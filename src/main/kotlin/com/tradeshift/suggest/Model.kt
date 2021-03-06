package com.tradeshift.suggest

import com.tradeshift.suggest.features.Inputs
import com.tradeshift.suggest.features.Update
import com.tradeshift.suggest.features.WordCounter

import com.tradeshift.suggest.storage.ModelTableStorage
import com.tradeshift.suggest.storage.Storage
import kotlin.math.ln

class Model(
        private val storage: Storage = ModelTableStorage(),
        private val pseudoCount: Double = 1.0 // reference: https://www.wikiwand.com/en/Naive_Bayes_classifier
        ) {

    private fun logProbability(count: Int, cFC: Int, cC: Int, numOfUniqueFeature: Int): Double {
        return count.toDouble() * (ln(cFC + pseudoCount) - ln(cC + numOfUniqueFeature * pseudoCount))
    }
    /**
     * Predicts using naive bayes, e.g. p(y|x) = p(x|y)p(y)/p(x)
     *
     * @return predicted outcomes and their un-normalized log-probability, e.g. {"positive": -4.566621, "negative": -2.324455}
     */
    fun predict(inputs: Inputs): Map<String, Double> {
        val result = hashMapOf<String, Double>()

        for (outcome in storage.getAllClasses()) {
            val priorsCountOfClass = storage.getPriorsCountOfClass(outcome)
            val totalDataCount = storage.getTotalDataCount()

            var lp = 0.0

            for ((featureName, featureValue) in inputs.text) {
                val knownFeaturesInTable = storage.getKnownWords(featureName)
                val countOfUniqueWord = knownFeaturesInTable.size
                val countOfAllWordInClass = storage.getCountOfAllWordInClass(featureName, outcome)

                val wordCountsForCurrentFeature = WordCounter.countWords(featureValue)
                val knownFeatures = wordCountsForCurrentFeature.keys.intersect(knownFeaturesInTable)

                for (entry in wordCountsForCurrentFeature.entries) {
                    if (knownFeatures.contains(entry.key)) {
                        lp += calculateLogProbability(featureName, outcome, countOfUniqueWord, countOfAllWordInClass, entry.value, entry.key)
                    }
                }
            }

            for ((featureName, featureValue) in inputs.category) {
                val knownFeaturesInTable = storage.getKnownWords(featureName)
                val countOfUniqueWord = knownFeaturesInTable.size
                val countOfAllWordInClass = storage.getCountOfAllWordInClass(featureName, outcome)

                if (featureValue in knownFeaturesInTable) {
                    lp += calculateLogProbability(featureName, outcome, countOfUniqueWord, countOfAllWordInClass, 1, featureValue)
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
        val countOfWordInClass = storage.getCountOfWordInClass(featureName, outcome, word)

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
    fun add(update: Update) {
        return batchAdd(listOf(update))
    }


    /**
     * @param updates List of observed Updates
     */
    fun batchAdd(updates: List<Update>) {

        for (update in updates) {
            val outcome = update.outcome
            val input = update.inputs

            for ((featureName, featureValue) in input.text) { // multinomial situation
                storage.addOneToPriorsCountOfClass(outcome)
                storage.addOneToTotalDataCount()

                val wordCountsForCurrentFeature = WordCounter.countWords(featureValue)
                for ((word, count) in wordCountsForCurrentFeature) {
                    storage.addToCountOfWordInClass(featureName, outcome, word, count)
                    storage.addOneToCountOfAllWordInClass(featureName, outcome, count)
                }
            }
            for ((featureName, featureValue) in input.category) { // category situation
                storage.addOneToPriorsCountOfClass(outcome)
                storage.addOneToTotalDataCount()
                storage.addToCountOfWordInClass(featureName, outcome, featureValue, 1)
                storage.addOneToCountOfAllWordInClass(featureName, outcome, 1)
            }
        }

    }
}
