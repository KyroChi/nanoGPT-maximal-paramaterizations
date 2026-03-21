"""
Vendored from megatron.core.datasets.indexed_dataset
Install megatron-core to use SlimPajama dataset, or use OpenWebText (memmap) instead.
"""
# TODO: Vendor IndexedDataset from megatron-core, or install megatron-core
raise ImportError(
    "IndexedDataset requires megatron-core. Install it with: pip install megatron-core\n"
    "Or use the OpenWebText dataset which uses numpy memmap and has no extra dependencies."
)
