package com.tradeshift.suggest.features

data class Features(
        val map: Map<String, Feature> = mapOf()) // feature name to feature value map

data class Feature(
        val featureValue: String = "",
        val isText: Boolean = false
)