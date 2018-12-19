package com.tradeshift.suggest.storage

operator fun <K1, K2, V> MutableMap<Pair<K1, K2>, V>.get(outcome: K1, feature: K2) = get(outcome to feature)
fun <K1, K2, V> MutableMap<Pair<K1, K2>, V>.put(outcome: K1, feature: K2, value: V) = put(outcome to feature, value)

typealias MutableTable<K1, K2, V> = MutableMap<Pair<K1, K2>, V>
typealias Outcome = String
typealias FeatureName = String
/*
 * A table to store count for a specific featureName
 */
data class ModelTable(
        val countOfWordInClass:  MutableTable<Outcome, String, Int> = hashMapOf<Pair<String, String>, Int>().withDefault { 0 },
        val countOfAllWordInClass: MutableMap<Outcome, Int> = hashMapOf(),
        val wordsAppeared: MutableSet<String> = mutableSetOf()
)

class ModelTableStorage(
        private val priorsCountOfClass: MutableMap<Outcome, Int> = hashMapOf(),
        private var totalDataCount: Int = 0,
        private val modelTableMapOfFeature: MutableMap<FeatureName, ModelTable> = hashMapOf()
) : Storage {
    override fun getAllClasses(): List<String> {
        return priorsCountOfClass.keys.toList()
    }

    override fun addOneToPriorsCountOfClass(c: String, value: Int ) {
        val count = priorsCountOfClass[c] ?: 0
        priorsCountOfClass.put(c, count + value)
    }

    override fun getPriorsCountOfClass(c: String): Int {
        return priorsCountOfClass[c] ?: 0
    }

    override fun addOneToTotalDataCount(value: Int ) {
        totalDataCount += value
    }

    override fun getTotalDataCount(): Int {
        return totalDataCount
    }

    override fun addToCountOfWordInClass(featureName: String, c: String, word: String, increment: Int) {
        val modelTable = modelTableMapOfFeature.getOrPut(featureName, { ModelTable() })
        val count = modelTable.countOfWordInClass[c, word] ?: 0
        modelTable.countOfWordInClass.put(c, word, count + increment)

        // add this word with this feature as well, no matter which class it is
        modelTable.wordsAppeared.add(word)
    }

    override fun getCountOfWordInClass(featureName: String, c: String, word: String): Int {
        val modelTable = modelTableMapOfFeature[featureName] ?: return 0
        return modelTable.countOfWordInClass[c, word] ?: 0
    }

    override fun addOneToCountOfAllWordInClass(featureName: String, c: String, value: Int) {
        val modelTable = modelTableMapOfFeature.getOrPut(featureName, { ModelTable() })
        val count = modelTable.countOfAllWordInClass[c] ?: 0
        modelTable.countOfAllWordInClass.put(c, count + value)
    }

    override fun getCountOfAllWordInClass(featureName: String, c: String): Int {
        val modelTable = modelTableMapOfFeature[featureName] ?: return 0
        return modelTable.countOfAllWordInClass[c] ?: 0
    }

    // do not call .toSet() in the end, otherwise predict on batches get very slow...
    override fun getKnownWords(featureName: String): Set<String> {
        val modelTable = modelTableMapOfFeature[featureName] ?: ModelTable()
        return modelTable.wordsAppeared
    }
}
