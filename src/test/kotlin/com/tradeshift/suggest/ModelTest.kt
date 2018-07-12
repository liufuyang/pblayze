package com.tradeshift.suggest

import com.tradeshift.suggest.features.Feature
import com.tradeshift.suggest.features.Features
import com.tradeshift.suggest.storage.ModelTableStorage
import com.tradeshift.suggest.storage.Outcome
import org.junit.Assert.assertEquals
import org.junit.Test
import kotlin.streams.toList
import kotlin.test.assertTrue


class ModelTest {

    @Test
    fun can_fit_20newsgroup() {
        val train = newsgroup("20newsgroup_train.txt")
        val model = Model()
        model.batchAdd(train)

        val test = newsgroup("20newsgroup_test.txt")
        val acc = test
                .parallelStream()
                .map {
                    if (it.first == model.predict(it.second).maxBy { it.value }?.key) {
                        1.0
                    } else {
                        0.0
                    }
                }
                .toList()
                .average()

        println(acc)
        assertTrue(abs(acc - 0.6438) < 0.00001) // sklearn MultinomialNB with a CountVectorizer gets ~0.646
    }


        @Test
    @Throws(Exception::class)
    fun batchAdd() {
        val model = Model()
        model.batchAdd(
            listOf("positive", "negative", "negative") zip
            listOf(
                    Features(map = mapOf(Pair("q", Feature("foo bar baz", true)))), //
                    Features(map = mapOf(Pair("q", Feature("foo foo bar baz zap zoo", true)))),
                    Features(map = mapOf(Pair("q", Feature("map pap mee zap", true))))
            )
        )
        val predictions = model.predict(Features(map = mapOf(Pair("q", Feature("foo")))))

        assertEquals(predictions.keys, setOf("positive", "negative"))

        val nUniqueWords = 8.0
        val nPositiveWords = 3.0
        val nNegativeWords = 10.0
        val nFooPositive = 1.0
        val nFooNegative = 2.0
        val nPositive = 1.0
        val nNegative = 2.0
        val nPseudoCount = 1.0
        val up = nPositive / (nPositive + nNegative) * ((nFooPositive + nPseudoCount) / (nPositiveWords + nUniqueWords))
        val un = nNegative / (nPositive + nNegative) * ((nFooNegative + nPseudoCount) / (nNegativeWords + nUniqueWords))
        val p = up / (up + un)
        val n = 1 - p

        assertEquals(p, predictions["positive"]!!, 0.000001)
        assertEquals(n, predictions["negative"]!!, 0.000001)
    }

    @Test
    @Throws(Exception::class)
    fun batchAddTwice() {

        val model1 = Model()
        model1.batchAdd(
            listOf("positive", "negative") zip
            listOf(
                    Features(map = mapOf(Pair("q", Feature("foo bar baz", true)))),
                    Features(map = mapOf(Pair("q", Feature("foo foo bar baz zap zoo", true))))
            ))

        model1.batchAdd(
                    listOf("negative") zip
                    listOf(
                            Features(map = mapOf(Pair("q", Feature("map pap mee zap", true))))
                    ))
        val predictions = model1.predict(Features(map = mapOf(Pair("q", Feature("foo", true)))))

        assertEquals(predictions.keys, setOf("positive", "negative"))

        val nUniqueWords = 8.0
        val nPositiveWords = 3.0
        val nNegativeWords = 10.0
        val nFooPositive = 1.0
        val nFooNegative = 2.0
        val nPositive = 1.0
        val nNegative = 2.0
        val nPseudoCount = 1.0
        val up = nPositive / (nPositive + nNegative) * ((nFooPositive + nPseudoCount) / (nPositiveWords + nUniqueWords))
        val un = nNegative / (nPositive + nNegative) * ((nFooNegative + nPseudoCount) / (nNegativeWords + nUniqueWords))
        val p = up / (up + un)
        val n = 1 - p

        assertEquals(p, predictions["positive"]!!, 0.000001)
        assertEquals(n, predictions["negative"]!!, 0.000001)
    }

    @Test
    @Throws(Exception::class)
    fun categoricalOnly() {
        val suggestions = model.predict(Features(map = mapOf(Pair("user", Feature("ole")))))

        /*
        Expected probabilities:

        P(p) = 1/4
        P(n) = 3/4
        From https://en.wikipedia.org/wiki/Categorical_distribution#Posterior_predictive_distribution
        and p has seen {ole} and n has seen {ole, bob, ada}

        posterior P(ole | p) = (1 + 1) / (1 + 3) = 1 / 2
        posterior P(ole | n) = (1 + 1) / (3 + 3) = 1 / 3

        P(p | ole) = P(p) * P(ole | p) / Norm = (1/4 * 1/2) / ((1/4 * 1/2) + ((3/4) * (1/3))) = 1/3
        P(n | ole) = P(n) * P(ole | n) / Norm = (3/4 * 1/3) / ((1/4 * 1/2) + ((3/4) * (1/3))) = 2/3
        */
        assertEquals(setOf("p", "n"), suggestions.keys)

        assertEquals(setOf("p", "n"), suggestions.keys)

        assertEquals(0.333333333, suggestions["p"]!!, 0.0000001)
        assertEquals(0.666666666, suggestions["n"]!!, 0.0000001)
    }

    @Test
    @Throws(Exception::class)
    fun multinomialOnly() {
        val suggestions = model.predict(Features(map = mapOf(Pair("q", Feature("awesome awesome awesome ok", true)))))

        // [[0.9268899 0.0731101]] from sklearn
        assertEquals(setOf("p", "n"), suggestions.keys)

        assertEquals(setOf("p", "n"), suggestions.keys)

        assertEquals(0.9268899, suggestions["p"]!!, 0.0000001)
        assertEquals(0.0731101, suggestions["n"]!!, 0.0000001)
    }

    @Test
    @Throws(Exception::class)
    fun emptyFeaturesDefaultToPrior() {
        val suggestions = model.predict(Features())

        assertEquals(setOf("p", "n"), suggestions.keys)
        assertEquals(setOf("p", "n"), suggestions.keys)

        assertEquals(0.25, suggestions["p"]!!, 0.0000001)
        assertEquals(0.75, suggestions["n"]!!, 0.0000001)
    }

    @Test
    @Throws(Exception::class)
    fun unseenFeaturesDefaultToPrior() {
        val suggestions = model.predict(Features(
                mapOf(
                        Pair("q", Feature("k k k k k k k k k k k k k k k k k", true)),
                        Pair("user", Feature("notseen")))
        ))
        assertEquals(setOf("p", "n"), suggestions.keys)

        assertEquals(0.25, suggestions["p"]!!, 0.0000001)
        assertEquals(0.75, suggestions["n"]!!, 0.0000001)
    }

    @Test
    fun stringWithSpecialChars() {
        val suggestions = model.predict(Features(map = mapOf(Pair("q", Feature("awesome.!!    awesome;;;awesome \t\n ok", true)))))

        // [[0.9268899 0.0731101]] from sklearn
        assertEquals(setOf("p", "n"), suggestions.keys)

        assertEquals(setOf("p", "n"), suggestions.keys)

        assertEquals(0.9268899, suggestions["p"]!!, 0.0000001)
        assertEquals(0.0731101, suggestions["n"]!!, 0.0000001)
    }

    private val model: Model
        get() {

            val modelTableStorage = ModelTableStorage()
            modelTableStorage.addOneToPriorsCountOfClass("p", 1)
            modelTableStorage.addOneToPriorsCountOfClass("n", 3)
            modelTableStorage.addOneToTotalDataCount(4)

            modelTableStorage.addToCountOfWordInClass("q", "p", "awesome", 7)
            modelTableStorage.addToCountOfWordInClass("q", "p", "terrible", 3)
            modelTableStorage.addToCountOfWordInClass("q", "p", "ok", 19)
            modelTableStorage.addToCountOfWordInClass("q", "n", "awesome", 2)
            modelTableStorage.addToCountOfWordInClass("q", "n", "terrible", 13)
            modelTableStorage.addToCountOfWordInClass("q", "n", "ok", 21)

            modelTableStorage.addOneToCountOfAllWordInClass("q", "p", 29)
            modelTableStorage.addOneToCountOfAllWordInClass("q", "n", 36)

            modelTableStorage.addToCountOfWordInClass("user", "p", "ole", 1)
            modelTableStorage.addToCountOfWordInClass("user", "n", "ole", 1)
            modelTableStorage.addToCountOfWordInClass("user", "n", "bob", 1)
            modelTableStorage.addToCountOfWordInClass("user", "n", "ada", 1)

            modelTableStorage.addOneToCountOfAllWordInClass("user", "p", 1)
            modelTableStorage.addOneToCountOfAllWordInClass("user", "n", 3)

            return Model(modelTableStorage)
        }

    private fun newsgroup(fname: String): List<Pair<Outcome, Features>> {
        val lines = this::class.java.classLoader.getResource(fname).readText(Charsets.UTF_8).split("\n")
        val features = mutableListOf<Pair<Outcome,Features>>()

        for (line in lines) {
            val split = line.split(" ".toRegex(), 2).toTypedArray()
            val outcome = split[0]
            var f = Features()
            if (split.size == 2) { //some are legit empty
                f = Features(mapOf(Pair("q", Feature(split[1], isText = true))))
            }
            features.add(outcome to f)
        }
        return features
    }

}
