package com.tradeshift.suggest.features

typealias FeatureName = String
typealias FeatureValue = String
typealias Outcome = String


data class Inputs(
        val text: Map<FeatureName, FeatureValue> = mapOf(),
        val category: Map<FeatureName, FeatureValue> = mapOf()
)

data class Update(
        val inputs: Inputs,
        val outcome: Outcome
)

fun inputOfText(vararg text: Pair<FeatureName, FeatureValue>) = Inputs(text = text.toMap())
fun inputOfCategory(vararg category: Pair<FeatureName, FeatureValue>) = Inputs(category = category.toMap())
