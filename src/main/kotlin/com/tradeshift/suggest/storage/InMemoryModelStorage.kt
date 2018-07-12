package com.tradeshift.suggest.storage

/**
 * A simple in-memory model store for testing purposes.
 */
class InMemoryModelStorage(
        private val modelStore: MutableMap<String, ModelTableStorage> = hashMapOf()) : ModelStorage {

    override fun alwaysGet(id: String): ModelTableStorage {
        val modelTableStorage = modelStore[id] ?: ModelTableStorage()
        modelStore.put(id, modelTableStorage)
        return modelStore[id]!!
    }

    override fun put(id: String, modelStorage: ModelTableStorage) {
        modelStore.put(id, modelStorage)
    }

}
