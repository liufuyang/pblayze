package com.tradeshift.suggest.storage

/**
 * Handles persistent storage of models.
 */
interface ModelStorage {
    fun alwaysGet(id: String): ModelTableStorage

    fun put(id: String, modelStorage: ModelTableStorage)
}
