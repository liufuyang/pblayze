package com.tradeshift.suggest

import com.tradeshift.suggest.features.Features
import com.tradeshift.suggest.features.WordCounter

import com.tradeshift.suggest.storage.ModelTableStorage
import com.tradeshift.suggest.storage.Outcome
import kotlin.math.ln

val pseudoCount = 1.0 // reference: https://www.wikiwand.com/en/Naive_Bayes_classifier

class Model(
        private val modelTableStorage: ModelTableStorage = ModelTableStorage()) {

    private fun logProbability(count: Int, cFC: Int, cC: Int, numOfUniqueFeature: Int): Double {
        return count.toDouble() * (ln(cFC + pseudoCount) - ln(cC + numOfUniqueFeature * pseudoCount))
    }
    /**
     * Predicts using naive bayes, e.g. p(y|x) = p(x|y)p(y)/p(x)
     *
     * @return predicted outcomes and their un-normalized log-probability, e.g. {"positive": -4.566621, "negative": -2.324455}
     */
    fun predict(features: Features): Map<String, Double> {
        val result = mutableMapOf<String, Double>()

        for (outcome in modelTableStorage.getAllClasses()) {
            val priorsCountOfClass = modelTableStorage.getPriorsCountOfClass(outcome)
            val totalDataCount = modelTableStorage.getTotalDataCount()

            var lp = 0.0

            for ((featureName, feature) in features.map) {
                val knownFeaturesInTable = modelTableStorage.getKnownWords(featureName)
                val countOfUniqueWord = knownFeaturesInTable.size
                val countOfAllWordInClass = modelTableStorage.getCountOfAllWordInClass(featureName, outcome)

                if (feature.isText) {
                    val wordCountsForCurrentFeature = WordCounter.countWords(feature.featureValue)
                    val knownFeatures = wordCountsForCurrentFeature.keys.intersect(knownFeaturesInTable)

                    for (word in knownFeatures) {
                        val count = wordCountsForCurrentFeature[word]
                        lp += calculateLogProbability(featureName, outcome, countOfUniqueWord, countOfAllWordInClass, count!!, word)
                    }
                } else {
                    if (feature.featureValue in knownFeaturesInTable) {
                        lp += calculateLogProbability(featureName, outcome, countOfUniqueWord, countOfAllWordInClass, 1, feature.featureValue)
                    }
                }
            }
            val finalLogP = ln(priorsCountOfClass.toDouble()) - ln(totalDataCount.toDouble()) + lp
            result.put(outcome, finalLogP)
        }

        return normalize(result)
    }

    private fun calculateLogProbability(featureName: String, outcome: String,
                                        countOfUniqueWord: Int,
                                        countOfAllWordInClass: Int,
                                        countOfWord: Int,
                                        word: String): Double {
        val countOfWordInClass = modelTableStorage.getCountOfWordInClass(featureName, outcome, word)

        return logProbability(countOfWord, countOfWordInClass, countOfAllWordInClass, countOfUniqueWord)
    }

    private fun normalize(suggestions: Map<String, Double>): Map<String, Double> {
        val max: Double = suggestions.maxBy({ it.value })?.value ?: 0.0
        val vals = suggestions.mapValues { Math.exp(it.value - max) }
        val norm = vals.values.sum()
        return vals.mapValues { it.value / norm }
    }

    /**
     * Creates a new model with the counts added
     */
    fun add(outcome: String, features: Features) {
        return batchAdd(listOf(outcome) zip listOf(features))
    }


    /**
     * @param outcomes List of observed Outcomes
     * @param features List of observed Features that corresponding to the outcomes
     */
    fun batchAdd(outcomeAndFeaturePairs: List<Pair<Outcome, Features>>) {

        for ((c, _features) in outcomeAndFeaturePairs) {
            for ((featureName, feature) in _features.map) {

                modelTableStorage.addOneToPriorsCountOfClass(c)
                modelTableStorage.addOneToTotalDataCount()

                if (feature.isText) { // multinomial situation
                    val wordCountsForCurrentFeature = WordCounter.countWords(feature.featureValue)
                    for ((word, count) in wordCountsForCurrentFeature) {
                        modelTableStorage.addToCountOfWordInClass(featureName, c, word, count)
                        modelTableStorage.addOneToCountOfAllWordInClass(featureName, c, count)
                    }
                } else { // categorical situation
                    modelTableStorage.addToCountOfWordInClass(featureName, c, feature.featureValue)
                    modelTableStorage.addOneToCountOfAllWordInClass(featureName, c)
                }
            }
        }

    }
}
