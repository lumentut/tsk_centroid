import numpy as np
import pandas as pd
import inspect
from typing import Union, Dict
from abc import ABC
from enum import Enum
from .fcm_mfs_cluster import FcmMFsCluster
from .kmeans_mfs_cluster import KMeansMFsCluster
from .mbk_mfs_cluster import MBKMFsCluster
from .fcm_rule_cluster import FcmRuleCluster
from .kmeans_rule_cluster import KMeansRuleCluster
from .mbk_rule_cluster import MBKRuleCluster
from src.utils.timing import Timing, execution_time


MFsCluster = Union[KMeansMFsCluster, FcmMFsCluster, MBKMFsCluster]
RulesCluster = Union[KMeansRuleCluster, FcmRuleCluster, MBKRuleCluster]


class ClusteringMethod(Enum):
    KMEANS = "kmeans"
    FUZZY_C_MEANS = "fuzzy_c_means"
    MBKMEANS = "mbkmeans"


class ClusterFactory(ABC):
    def create_clusterer(
        self, method: ClusteringMethod, **kwargs
    ) -> Union[MFsCluster, RulesCluster]:
        strategy_class = self._strategies.get(method)
        if strategy_class is not None:
            clusterer_kwargs = {}
            for k, v in kwargs.items():
                if k.startswith(self._kwargs_prefix):
                    new_key = k[len(self._kwargs_prefix) :]
                    clusterer_kwargs[new_key] = v

            signature_param_names = [
                param.name
                for param in inspect.signature(
                    strategy_class.__init__
                ).parameters.values()
                if param.name != "self"
            ]
            strategy_kwargs = {
                k: v for k, v in clusterer_kwargs.items() if k in signature_param_names
            }
            return strategy_class(**strategy_kwargs)
        raise ValueError(f"Clustering method '{method}' is not supported.")


class MFsClusterFactory(ClusterFactory):
    _strategies = {
        ClusteringMethod.KMEANS: KMeansMFsCluster,
        ClusteringMethod.FUZZY_C_MEANS: FcmMFsCluster,
        ClusteringMethod.MBKMEANS: MBKMFsCluster,
    }

    _kwargs_prefix = "mfs__cluster__"


class RulesClusterFactory(ClusterFactory):
    _strategies = {
        ClusteringMethod.KMEANS: KMeansRuleCluster,
        ClusteringMethod.FUZZY_C_MEANS: FcmRuleCluster,
        ClusteringMethod.MBKMEANS: MBKRuleCluster,
    }

    _kwargs_prefix = "rules__cluster__"


class ClusterContext(Enum):
    MFS = "mfs"
    RULES = "rules"


class ClusterFactorySelector:
    _factories = {
        ClusterContext.MFS: MFsClusterFactory(),
        ClusterContext.RULES: RulesClusterFactory(),
    }

    @classmethod
    def get_factory(cls, context: ClusterContext) -> ClusterFactory:
        factory = cls._factories.get(context)
        if factory is None:
            raise ValueError(f"Factory '{context}' is not registered.")
        return factory


class Cluster:
    def __init__(
        self,
        cluster_ctx: ClusterContext,
        method: ClusteringMethod,
        **kwargs,
    ):
        self.cluster_factory = ClusterFactorySelector.get_factory(cluster_ctx)
        self.strategy = self.cluster_factory.create_clusterer(method, **kwargs)

    def fit(self, data: Union[Dict, pd.DataFrame, np.ndarray]):
        return self.strategy.fit(data)


class Clusters(Timing):
    def __init__(self, df: pd.DataFrame, method: ClusteringMethod, **kwargs):
        self.df = df.copy()
        self.method = method
        self.mfs_clusters_ = self._create_mfs_clusters(**kwargs)
        self.rules_cluster_ = self._create_rules_cluster(**kwargs)

    def _create_mf_cluster(self, values: np.ndarray, **kwargs) -> MFsCluster:
        cluster_factory = ClusterFactorySelector.get_factory(context=ClusterContext.MFS)
        strategy = cluster_factory.create_clusterer(self.method, **kwargs)
        strategy.fit(values)
        return strategy

    @execution_time
    def _create_mfs_clusters(self, **kwargs) -> dict[str, MFsCluster]:
        mf_clusters_ = {}
        features = self.df.columns.tolist()
        for feature in features:
            values = self.df[feature].to_numpy()
            mf_cluster = self._create_mf_cluster(values, **kwargs)
            mf_clusters_[feature] = mf_cluster
        return mf_clusters_

    @execution_time
    def _create_rules_cluster(self, **kwargs) -> RulesCluster:
        cluster_factory = ClusterFactorySelector.get_factory(
            context=ClusterContext.RULES
        )
        strategy = cluster_factory.create_clusterer(self.method, **kwargs)
        strategy.fit(self.df)
        return strategy
