import com.tradeshift.suggest.Model
import com.tradeshift.suggest.features.Inputs
import com.tradeshift.suggest.features.Update
import com.tradeshift.suggest.storage.ModelTableStorage
import org.junit.Test
import kotlin.streams.toList

class ProductClassificationTest {

    @Test
    fun can_fit_product_classification() {
        val train = productClassification("train.csv")
        val model = Model(ModelTableStorage(), pseudoCount = 0.01)
        model.batchAdd(train)

        println("training finished")
        val test = productClassification("test.csv")
        val acc = test
                .parallelStream()
                .map {
                    if (it.outcome == model.predict(it.inputs).maxBy { it.value }?.key) {
                        1.0
                    } else {
                        0.0
                    }
                }
                .toList()
                .average()

        println(acc)
    }

    fun productClassification(fname: String): List<Update> {
        val lines = this::class.java.getResource(fname).readText(Charsets.UTF_8).split("\n")
        val updates = mutableListOf<Update>()

        for (line in lines) {
            val outcome = line.substringAfterLast(',')
            val input = line.substringBeforeLast(',')
            var f = Inputs(mapOf("q" to input))
            updates.add(Update(f, outcome))
        }
        return updates
    }
}

