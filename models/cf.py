import os
from math import log
from operator import mul
from functools import reduce
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.mllib.linalg.distributed import CoordinateMatrix
from pyspark.sql.functions import udf, explode, pandas_udf, PandasUDFType, array, slice, array_union, col, collect_list
from pyspark.sql.functions import sum as sql_sum, count as sql_count, avg as sql_avg, struct
from pyspark.sql.types import IntegerType, FloatType, StringType, ArrayType, StructType, StructField
from tools.utils import dict_get_keys


class KnnBase:
    def __init__(self, config):
        """
        :param config: json
        for example:
        {
            "rec_list_len": 100,
            "params":{
                "k": 30,
                "sim_type": "cosine"
            }
        }
        """
        print(str(self))
        self.model_config = config
        self.params = config.get("params")

    def train_model(self, score_indexed_df):
        """
        :param score_indexed_df: dataframe [uid_index, item_index, score]
        """
        sim_type = self.params.get("sim_type")
        if sim_type == 'cosine':
            mat = CoordinateMatrix(score_indexed_df.rdd)
            row_matrix = mat.toRowMatrix()
            sims = row_matrix.columnSimilarities()

            def get_sim_symmetric(entries):
                # cols: item, item_sim, sim
                result = []
                for entry in entries:
                    result.append(Row(item=entry.i, item_sim=entry.j, sim=entry.value))
                    result.append(Row(item=entry.j, item_sim=entry.i, sim=entry.value))
                return iter(result)

            self.item_sim_df = sims.entries.mapPartitions(get_sim_symmetric).toDF()
        else:
            raise NotImplementedError("sim_type %s is not implemented yet" % sim_type)

    def generate_recommend_list_df(self, score_indexed_df, uid_indexed_df, item_indexed_df,
                                   spark, item_sim_df=None):
        pass


class KnnWithMeans(KnnBase):

    def __str__(self):
        return "Init KnnWithMeans"

    def generate_recommend_list_df(self, score_indexed_df, uid_indexed_df, item_indexed_df,
                                       spark, item_sim_df=None):
            """
            :param score_indexed_df: dataframe with columns: [uid_index, item_index, score]
            :param uid_indexed_df: dataframe with columns: [uid, uid_index]
            :param item_indexed_df: dataframe with columns: [item, item_index]
            :param spark: instance of SparkSession
            :return: dataframe [uid_index, item_index, rating]
            """
            # item mean score
            @pandas_udf(FloatType(), PandasUDFType.GROUPED_AGG)
            def mean_udf(v):
                return v.mean()
            mean_df = score_indexed_df.select("item_index", "score").groupby("item_index").agg(
                mean_udf("score").alias("score_mean"))
            mean_df = mean_df.cache()

            select_uid_df = uid_indexed_df.select("uid_index")

            score_indexed_df = score_indexed_df.join(select_uid_df,
                                                     score_indexed_df["uid_index"] == select_uid_df["uid_index"],
                                                     "right_outer").drop(score_indexed_df.uid_index)
            if item_sim_df is None:
                item_sim_df = self.item_sim_df
            train_indexed_mean_df = score_indexed_df.na.fill({"item_index": -1})

            ### broadcast item_sim, item_mean
            item_sim_broadcast = spark.sparkContext.broadcast({str(row["item"]) + "," + str(row["item_sim"]): row["sim"] for row in item_sim_df.collect()})
            item_mean_broadcast = spark.sparkContext.broadcast({row["item_index"]: row["score_mean"] for row in mean_df.collect()})
            item_index_broadcast = spark.sparkContext.broadcast(set([row["item_index"] for row in item_indexed_df.select("item_index").collect()]))

            train_indexed_mean_df = train_indexed_mean_df.select("uid_index", array("item_index", "score").alias("item_score_h"))
            train_indexed_mean_df = train_indexed_mean_df.groupby("uid_index").agg(collect_list("item_score_h").alias("item_score_h"))

            k = self.params.get("k", 30)

            def rec_for_uid(item_score_h):
                item_mean_dict = item_mean_broadcast.value
                item_sim_dict = item_sim_broadcast.value
                item_index_set = item_index_broadcast.value
                result = []
                for item_pred in item_index_set:
                    item_pred_score_mean = item_mean_dict[item_pred]
                    ups = []
                    for item, score in item_score_h:
                        if item == -1:
                            # indicate this uid has no history
                            ups.append([0., 0.])
                            continue
                        key = str(item_pred)+","+str(int(item))
                        if key in item_sim_dict:
                            item_h_score_mean = item_mean_dict[item]
                            ups.append([item_sim_dict[key], float(score-item_h_score_mean)])
                    ups = sorted(ups, key=lambda x: x[0], reverse=True)[:k]
                    up = sum([x[0] * x[1] for x in ups])
                    dn = sum([x[0] for x in ups])
                    rating = item_pred_score_mean
                    if dn != 0:
                        rating += up * 1. / dn
                    result.append([item_pred, rating])
                return result

            rec_for_uid_udf = udf(rec_for_uid, ArrayType(
                        StructType([StructField("item_index", IntegerType()), StructField("rating", FloatType())])))
            pred_indexed_df = train_indexed_mean_df.withColumn("uid_item_pred_list", rec_for_uid_udf("item_score_h"))
            pred_indexed_df = pred_indexed_df.select("uid_index", explode("uid_item_pred_list").alias("uid_item_pred_list"))
            pred_indexed_df = pred_indexed_df.select("uid_index", "uid_item_pred_list.item_index", "uid_item_pred_list.rating")

            return pred_indexed_df


class KnnBasic(KnnBase):

    def __str__(self):
        return "Init KnnBasic"

    def generate_recommend_list_df(self, score_indexed_df, uid_indexed_df, item_indexed_df,
                                       spark, item_sim_df=None):
            """
            :param score_indexed_df: dataframe with columns: [uid_index, item_index, score]
            :param uid_indexed_df: dataframe with columns: [uid, uid_index]
            :param item_indexed_df: dataframe with columns: [item, item_index]
            :param spark: instance of SparkSession
            :return: dataframe [uid_index, item_index, rating]
            """
            # item mean score
            @pandas_udf(FloatType(), PandasUDFType.GROUPED_AGG)
            def mean_udf(v):
                return v.mean()
            mean_df = score_indexed_df.select("item_index", "score").groupby("item_index").agg(
                mean_udf("score").alias("score_mean"))
            mean_df = mean_df.cache()

            select_uid_df = uid_indexed_df.select("uid_index")

            score_indexed_df = score_indexed_df.join(select_uid_df,
                                                     score_indexed_df["uid_index"] == select_uid_df["uid_index"],
                                                     "right_outer").drop(score_indexed_df.uid_index)
            if item_sim_df is None:
                item_sim_df = self.item_sim_df
            train_indexed_mean_df = score_indexed_df.na.fill({"item_index": -1})

            ### broadcast item_sim, item_mean
            item_sim_broadcast = spark.sparkContext.broadcast({str(row["item"]) + "," + str(row["item_sim"]): row["sim"] for row in item_sim_df.collect()})
            item_mean_broadcast = spark.sparkContext.broadcast({row["item_index"]: row["score_mean"] for row in mean_df.collect()})
            item_index_broadcast = spark.sparkContext.broadcast(set([row["item_index"] for row in item_indexed_df.select("item_index").collect()]))

            train_indexed_mean_df = train_indexed_mean_df.select("uid_index", array("item_index", "score").alias("item_score_h"))
            train_indexed_mean_df = train_indexed_mean_df.groupby("uid_index").agg(collect_list("item_score_h").alias("item_score_h"))

            k = self.params.get("k", 30)

            def rec_for_uid(item_score_h):
                """is_basic indicate knn without item mean score, else use item mean score"""
                item_mean_dict = item_mean_broadcast.value
                item_sim_dict = item_sim_broadcast.value
                item_index_set = item_index_broadcast.value
                result = []
                for item_pred in item_index_set:
                    item_pred_score_mean = item_mean_dict[item_pred]
                    ups = []
                    for item, score in item_score_h:
                        if item == -1:
                            # indicate this uid has no history
                            ups.append([0., 0.])
                            continue
                        key = str(item_pred)+","+str(int(item))
                        if key in item_sim_dict:
                            ups.append([item_sim_dict[key], float(score)])

                    ups = sorted(ups, key=lambda x: x[0], reverse=True)[:k]
                    up = sum([x[0] * x[1] for x in ups])
                    dn = sum([x[0] for x in ups])
                    rating = 0.
                    if dn != 0:
                        rating += up * 1. / dn
                    else:
                        rating += item_pred_score_mean

                    result.append([item_pred, rating])
                return result

            rec_for_uid_udf = udf(rec_for_uid, ArrayType(
                        StructType([StructField("item_index", IntegerType()), StructField("rating", FloatType())])))
            pred_indexed_df = train_indexed_mean_df.withColumn("uid_item_pred_list", rec_for_uid_udf("item_score_h"))
            pred_indexed_df = pred_indexed_df.select("uid_index", explode("uid_item_pred_list").alias("uid_item_pred_list"))
            pred_indexed_df = pred_indexed_df.select("uid_index", "uid_item_pred_list.item_index", "uid_item_pred_list.rating")

            return pred_indexed_df
