package com.tradeshift.suggest.storage

operator fun <K1, K2, V> MutableMap<Pair<K1, K2>, V>.get(outcome: K1, feature: K2) = get(outcome to feature)
//operator fun <K1, K2, V> MutableMap<Pair<K1, K2>, V>.put(outcome: K1, feature: K2, value: V) = put(outcome to feature, value)

typealias MutableTable<K1, K2, V> = MutableMap<Pair<K1, K2>, V>
typealias Outcome = String
typealias FeatureName = String
/*
 * A table to store count for a specific featureName
 */
data class ModelTable(
        val countOfWordInClass:  MutableTable<Outcome, FeatureName, Int> = mutableMapOf<Pair<String, String>, Int>().withDefault { 0 },
        val countOfAllWordInClass: MutableMap<String, Int> = mutableMapOf(),
        val wordsAppeared: MutableSet<String> = mutableSetOf()
)

class ModelTableStorage(
        private val priorsCountOfClass: MutableMap<String, Int> = mutableMapOf(),
        private var totalDataCount: Int = 0,

        private val modelTableMapOfFeature: MutableMap<String, ModelTable> = mutableMapOf()
){
    fun getAllClasses(): List<String> {
        return priorsCountOfClass.keys.toList()
    }

    fun addOneToPriorsCountOfClass(c: String, value: Int = 1 ) {
        val count = priorsCountOfClass[c] ?: 0
        priorsCountOfClass.put(c, count + value)
    }

    fun getPriorsCountOfClass(c: String): Int {
        return priorsCountOfClass[c] ?: 0
    }

    fun addOneToTotalDataCount(value: Int = 1 ) {
        totalDataCount += value
    }

    fun getTotalDataCount(): Int {
        return totalDataCount
    }

    fun addToCountOfWordInClass(featureName: String, c: String, word: String, increment: Int = 1) {
        val modelTable = modelTableMapOfFeature[featureName] ?: ModelTable()
        modelTableMapOfFeature.put(featureName, modelTable)
        val count = modelTable.countOfWordInClass.get(c, word) ?: 0
        modelTable.countOfWordInClass.put(c to word, count + increment)

        // add this word with this feature as well, no matter which class it is
        modelTable.wordsAppeared.add(word)
    }

    fun getCountOfWordInClass(featureName: String, c: String, word: String): Int {
        val modelTable = modelTableMapOfFeature[featureName] ?: return 0
        return modelTable.countOfWordInClass[c, word] ?: 0
    }

    fun addOneToCountOfAllWordInClass(featureName: String, c: String, value: Int = 1) {
        val modelTable = modelTableMapOfFeature[featureName] ?: ModelTable()
        modelTableMapOfFeature.put(featureName, modelTable)
        val count = modelTable.countOfAllWordInClass[c] ?: 0
        modelTable.countOfAllWordInClass.put(c, count + value)
    }

    fun getCountOfAllWordInClass(featureName: String, c: String): Int {
        val modelTable = modelTableMapOfFeature[featureName] ?: return 0
        return modelTable.countOfAllWordInClass[c] ?: 0
    }

    // do not call .toSet() in the end, otherwise predict on batches get very slow...
    fun getKnownWords(featureName: String): Set<String> {
        val modelTable = modelTableMapOfFeature[featureName] ?: ModelTable()
        return modelTable.wordsAppeared
    }
}
