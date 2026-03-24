#ifndef TOPOLOGY_H
#define TOPOLOGY_H

/* topology.h — hardware topology and bandwidth measurements for BREAD.
 *
 * topology_probe() runs all benchmarks, analyses the GGUF model, prints
 * the scheduler summary, and returns the measured values for use by the
 * rest of the engine (loader, scheduler).
 */

#include <stdint.h>

typedef struct {
    /* ---- GPU ---- */
    char     gpu_name[256];
    uint64_t vram_total_mb;
    uint64_t vram_free_mb;          /* free at probe time */

    /* ---- System RAM ---- */
    uint64_t ram_total_mb;

    /* ---- Measured bandwidth ---- */
    double   bw_ram_to_vram_gbs;    /* peak across tested sizes, GB/s */
    double   bw_ssd_to_ram_gbs;     /* sustained sequential fread, GB/s */

    /* ---- Model weight classification (filled from GGUF) ---- */
    uint64_t model_total_bytes;
    uint64_t model_nonexpert_bytes; /* tensors without "*_exps*": pin in VRAM */
    uint64_t model_expert_bytes;    /* tensors with    "*_exps*": stream per layer */
    uint64_t n_moe_layers;          /* number of MoE transformer layers */
    uint64_t layer_expert_bytes;    /* expert weight bytes for one layer (3 tensors) */
} topo_t;

/* Probe hardware bandwidth, analyse the GGUF model at model_path,
 * print the topology summary, and return the measured values. */
topo_t topology_probe(const char *model_path);

#endif /* TOPOLOGY_H */
