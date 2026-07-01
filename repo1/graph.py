import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt

# ======================================================
# CONFIG — change only this section
# ======================================================

TIC_ID = "100100827"   # WASP-18 b test case
SECTOR = 2

SDE_THRESHOLD = 7.0
SNR_THRESHOLD = 7.0
ODD_EVEN_TOLERANCE = 0.5

# ======================================================
# STEP 1: SEARCH + DOWNLOAD
# ======================================================

print("Searching MAST for light curve...")

if SECTOR:
    search_result = lk.search_lightcurve(f"TIC {TIC_ID}", mission="TESS", sector=SECTOR)
else:
    search_result = lk.search_lightcurve(f"TIC {TIC_ID}", mission="TESS")

print(search_result)

if len(search_result) == 0:
    raise ValueError("No light curve found. Check TIC ID / sector.")

lc = search_result.download()
tic_id = lc.meta.get("OBJECT", TIC_ID)
print(f"\nLoaded light curve for TIC {tic_id}")

# ======================================================
# STEP 2: CLEAN + FLATTEN
# ======================================================

lc = lc.remove_nans().remove_outliers(sigma=5).normalize()
flat_lc = lc.flatten(window_length=901)

# ======================================================
# STEP 3: BLS SEARCH
# ======================================================

periods = np.linspace(0.5, 15, 8000)
durations = np.linspace(0.05, 0.3, 10)

bls = flat_lc.to_periodogram(method="bls", period=periods, duration=durations)

best_period = float(bls.period_at_max_power.value)
best_t0 = bls.transit_time_at_max_power
best_duration = float(bls.duration_at_max_power.value)  # in days

power = bls.power.value
sde = float((power.max() - np.median(power)) / np.std(power))

print("\n========== BLS RESULTS ==========")
print(f"Best Period   = {best_period:.4f} days")
print(f"Best Duration = {best_duration:.4f} days")
print(f"SDE           = {sde:.2f}")

# ======================================================
# STEP 4: PHASE FOLD + DEPTH / SNR
# NOTE: folded.phase.value is in DAYS (range -P/2 to +P/2),
# NOT a 0-1 fraction. All masks below use day units.
# ======================================================

folded = flat_lc.fold(period=best_period, epoch_time=best_t0)

half_dur = best_duration / 2.0

in_transit_mask = np.abs(folded.phase.value) < half_dur
out_transit_mask = ~in_transit_mask

flux_in = folded.flux.value[in_transit_mask]
flux_out = folded.flux.value[out_transit_mask]

baseline = float(np.nanmedian(flux_out))
transit_min = float(np.nanmedian(flux_in)) if len(flux_in) > 0 else baseline

depth_fraction = baseline - transit_min
depth_ppm = depth_fraction * 1e6

noise = float(np.nanstd(flux_out))
snr = depth_fraction / noise if noise > 0 else 0

rp_rs = np.sqrt(max(depth_fraction, 0))

print("\n========== TRANSIT METRICS ==========")
print(f"In-transit points  = {in_transit_mask.sum()}")
print(f"Out-transit points = {out_transit_mask.sum()}")
print(f"Depth   = {depth_ppm:.1f} ppm")
print(f"Rp/R*   = {rp_rs:.4f}")
print(f"SNR     = {snr:.2f}")

# ======================================================
# STEP 5: ODD-EVEN TRANSIT CHECK (false positive screen)
# ======================================================

time = flat_lc.time.value
flux = flat_lc.flux.value

phase_num = np.floor((time - best_t0.value) / best_period + 0.5)
folded_phase_days = (time - best_t0.value) - phase_num * best_period  # in DAYS

in_transit = np.abs(folded_phase_days) < half_dur

odd_mask = in_transit & (phase_num % 2 != 0)
even_mask = in_transit & (phase_num % 2 == 0)

odd_depth = baseline - np.nanmedian(flux[odd_mask]) if odd_mask.sum() > 3 else np.nan
even_depth = baseline - np.nanmedian(flux[even_mask]) if even_mask.sum() > 3 else np.nan

if np.isfinite(odd_depth) and np.isfinite(even_depth) and even_depth != 0:
    odd_even_ratio = abs(odd_depth - even_depth) / max(abs(even_depth), 1e-10)
else:
    odd_even_ratio = np.nan

print("\n========== ODD-EVEN CHECK ==========")
print(f"Odd-transit points  = {odd_mask.sum()}")
print(f"Even-transit points = {even_mask.sum()}")
print(f"Odd depth  = {odd_depth*1e6:.1f} ppm" if np.isfinite(odd_depth) else "Odd depth  = N/A")
print(f"Even depth = {even_depth*1e6:.1f} ppm" if np.isfinite(even_depth) else "Even depth = N/A")
print(f"Odd-Even fractional diff = {odd_even_ratio:.2f}" if np.isfinite(odd_even_ratio) else "Odd-Even fractional diff = N/A")

odd_even_ok = (not np.isfinite(odd_even_ratio)) or (odd_even_ratio < ODD_EVEN_TOLERANCE)

# ======================================================
# STEP 6: SECONDARY ECLIPSE CHECK
# Secondary eclipse sits at phase = +-P/2 in DAYS
# ======================================================

secondary_mask = np.abs(np.abs(folded.phase.value) - (best_period / 2)) < half_dur
secondary_flux = folded.flux.value[secondary_mask]
secondary_depth = baseline - np.nanmedian(secondary_flux) if len(secondary_flux) > 3 else 0

secondary_significant = secondary_depth > (3 * noise) if noise > 0 else False

print("\n========== SECONDARY ECLIPSE CHECK ==========")
print(f"Secondary points = {secondary_mask.sum()}")
print(f"Secondary depth = {secondary_depth*1e6:.1f} ppm")
print(f"Significant secondary eclipse: {secondary_significant}")

# ======================================================
# STEP 7: FINAL VERDICT
# ======================================================

checks_passed = (
    sde > SDE_THRESHOLD and
    snr > SNR_THRESHOLD and
    depth_fraction > 0 and
    odd_even_ok and
    not secondary_significant
)

verdict = "TRANSIT CANDIDATE" if checks_passed else "NO TRANSIT"

print("\n================ FINAL VERDICT ================")
print(f"TIC ID        : {tic_id}")
print(f"Period        : {best_period:.4f} days")
print(f"Depth         : {depth_ppm:.1f} ppm")
print(f"SDE           : {sde:.2f}  (threshold {SDE_THRESHOLD})")
print(f"SNR           : {snr:.2f}  (threshold {SNR_THRESHOLD})")
print(f"Odd-Even OK   : {odd_even_ok}")
print(f"No 2ndary ecl.: {not secondary_significant}")
print(f"VERDICT       : {verdict}")
print("================================================")

# ======================================================
# STEP 8: PLOTS
# ======================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

folded.scatter(ax=axes[0], xlabel="Phase (days)", ylabel="Normalized Flux")
axes[0].axvspan(-half_dur, half_dur, color="orange", alpha=0.2, label="in-transit window")
axes[0].legend()
axes[0].set_title(f"Phase-folded | P={best_period:.4f}d | {verdict}")

bls.plot(ax=axes[1])
axes[1].set_title(f"BLS Periodogram | SDE={sde:.2f}")

plt.tight_layout()
plt.show()

# ======================================================
# DATASET ROW (for CSV building)
# ======================================================

print("\n===== DATASET ENTRY (TSV) =====")
print(f"{tic_id}\t{best_period:.4f}\t{depth_ppm:.0f}\t{rp_rs:.4f}\t{snr:.2f}\t{sde:.2f}\t{verdict}")