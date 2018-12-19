package com.tradeshift.suggest.storage

interface Storage {
    fun getAllClasses(): List<String>
    fun addOneToPriorsCountOfClass(c: String, value: Int = 1 )
    fun getPriorsCountOfClass(c: String): Int
    fun addOneToTotalDataCount(value: Int = 1 )
    fun getTotalDataCount(): Int
    fun addToCountOfWordInClass(featureName: String, c: String, word: String, increment: Int = 1)
    fun getCountOfWordInClass(featureName: String, c: String, word: String): Int
    fun addOneToCountOfAllWordInClass(featureName: String, c: String, value: Int = 1)
    fun getCountOfAllWordInClass(featureName: String, c: String): Int
    // do not call .toSet() in the end, otherwise predict on batches get very slow...
    fun getKnownWords(featureName: String): Set<String>
}
