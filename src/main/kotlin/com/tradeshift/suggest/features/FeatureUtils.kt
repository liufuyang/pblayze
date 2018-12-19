package com.tradeshift.suggest.features

object WordCounter {

    private val stopWords = WordCounter::class.java.getResource("/data/english-stop-words-large.txt")
            .readText().split("\n").toSet()

    fun countWords(q: String): Map<String, Int> {
        // https://www.regular-expressions.info/unicode.html
        val returnString = q
                .replace("[^a-zA-Z]+".toRegex(), " ") // only keep any kind of letter from any language, others become space
                .trim()
                .toLowerCase()
                .split(" ")
                .filter { it !in stopWords }
        // with this setup, the 20newsgroup score is above 0.64

        return returnString.groupingBy { it }.eachCount()
    }
}
