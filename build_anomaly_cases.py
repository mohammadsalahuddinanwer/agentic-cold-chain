"""
Rebuilds anomaly_cases.json with:
1. Discriminative format - anomaly type and key signal at the FRONT
2. Balanced classes - 10 cases per evaluation class
3. Clear distinguishing language between similar classes


"""
import json
from pathlib import Path

# ── KEY DISTINGUISHING SIGNALS PER CLASS ─────────────────────────────────
# These are the phrases that make each class unique in the embedding space
DISCRIMINATORS = {
    "compressor_failure": "MECHANICAL FAILURE of refrigeration unit internals (compressor, fan, valve, refrigerant). Internal hardware fault. NOT caused by power loss or external weather.",
    "power_outage":       "TOTAL ELECTRICAL POWER LOSS to the vehicle or refrigeration unit. Mains/generator failure. All electrical systems affected. NOT a mechanical compressor fault.",
    "door_left_open":     "PHYSICAL DOOR OPEN or DOOR SEAL BREACH. Warm ambient air entering through open/ajar door or degraded seal. Driver or loading bay human action. Temperature rise is GRADUAL.",
    "driver_error":       "HUMAN OPERATOR MISTAKE. Driver set wrong temperature, ignored alarm, turned off unit by mistake, or failed to close door properly. DELIBERATE or NEGLIGENT human action.",
    "extreme_weather":    "EXTERNAL ENVIRONMENTAL CAUSE. Weather event (storm, heatwave, snow, rain) or extreme ambient temperature causing cooling system to struggle. No internal fault.",
    "gps_blackout":       "GPS SIGNAL LOSS ONLY. Location tracking unavailable. Temperature is STABLE and unaffected. No thermal anomaly. Navigation issue only.",
}

def fmt(anomaly_type, key_signal, description, severity, resolution, pattern, product_type=None):
    """Build a discriminative KB entry."""
    meta = {
        "type": anomaly_type,
        "severity": severity,
        "resolution": resolution,
        "pattern": pattern,
    }
    if product_type:
        meta["product_type"] = product_type

    text = (
        f"ANOMALY_TYPE: {anomaly_type}. "
        f"KEY_SIGNAL: {DISCRIMINATORS[anomaly_type]} "
        f"DESCRIPTION: {description} "
        f"RESOLUTION: {resolution}."
    )
    return {"text": text, "metadata": meta}


# ═══════════════════════════════════════════════════════════════════════════
# 10 × compressor_failure
# ═══════════════════════════════════════════════════════════════════════════
compressor = [
    fmt("compressor_failure", None,
        "Compressor seized; temperature spiked 3.5°C in 6 minutes. Refrigeration unit making grinding noise.",
        "high", "dispatch_backup", "compressor_seized"),

    fmt("compressor_failure", None,
        "Rapid temperature rise from 4°C to 8°C in under 8 minutes; refrigeration unit stopped completely with no power loss.",
        "high", "dispatch_backup", "refrigeration_unit_stopped"),

    fmt("compressor_failure", None,
        "Cooling unit malfunction: temperature rose 4°C to 9°C in 5 minutes. Compressor running but not cooling.",
        "high", "dispatch_backup", "cooling_unit_malfunction"),

    fmt("compressor_failure", None,
        "Refrigerant leak detected by sensor; temperature climbed 4°C to 7.5°C in 10 minutes. Hissing sound reported.",
        "high", "dispatch_backup", "refrigerant_leak"),

    fmt("compressor_failure", None,
        "Compressor overheating triggered thermal cutout; temperature rose 4°C to 8.2°C in 7 minutes. Unit restarted but failed again.",
        "high", "dispatch_backup", "compressor_overheating"),

    fmt("compressor_failure", None,
        "Evaporator fan failure: reduced airflow causing uneven cooling. Temperature gradually rising 0.3°C per minute.",
        "medium", "repair_on_site", "evaporator_fan_failure"),

    fmt("compressor_failure", None,
        "Defrost cycle stuck: ice build-up on evaporator blocked airflow. Temperature rising despite compressor running.",
        "medium", "repair_on_site", "defrost_cycle_failure"),

    fmt("compressor_failure", None,
        "Low refrigerant pressure: cooling efficiency reduced by 60%. Temperature slowly rising 0.2°C per minute over 40 minutes.",
        "medium", "repair_on_site", "low_refrigerant_pressure"),

    fmt("compressor_failure", None,
        "Expansion valve blocked; uneven cooling across cargo. Temperature fluctuating 2°C above setpoint.",
        "medium", "repair_on_site", "expansion_valve_failure"),

    fmt("compressor_failure", None,
        "Pharmaceutical shipment 2-8°C range. Compressor cycling failure caused temperature to reach 9°C. Internal mechanical fault confirmed.",
        "high", "dispatch_backup", "pharmaceutical_compressor_cycling",
        product_type="pharmaceuticals"),
]

# ═══════════════════════════════════════════════════════════════════════════
# 10 × power_outage
# ═══════════════════════════════════════════════════════════════════════════
power_outage = [
    fmt("power_outage", None,
        "Generator failed completely; all electrical systems including refrigeration lost power. Temperature rose 4°C to 10°C in 18 minutes.",
        "high", "dispatch_backup", "generator_failed"),

    fmt("power_outage", None,
        "Mains power failure caused refrigeration to stop; total electrical blackout on vehicle. Temperature rose 4°C to 9°C in 15 minutes.",
        "high", "dispatch_backup", "mains_power_failure"),

    fmt("power_outage", None,
        "Electrical outage: fuse blown, all power lost to refrigeration unit. Temperature rose 4°C to 8.5°C in 12 minutes.",
        "high", "dispatch_backup", "electrical_outage_fuse"),

    fmt("power_outage", None,
        "Power surge caused complete shutdown of refrigeration electronics. All systems offline. Temperature rose rapidly.",
        "high", "dispatch_backup", "power_surge_shutdown"),

    fmt("power_outage", None,
        "Battery bank depleted; vehicle running on reserve. Refrigeration unit lost power. Temperature rising 0.5°C per minute.",
        "high", "dispatch_backup", "battery_depleted"),

    fmt("power_outage", None,
        "Alternator failure: vehicle battery not charging. Refrigeration unit powered down after 20 minutes. Total electrical failure.",
        "high", "dispatch_backup", "alternator_failure"),

    fmt("power_outage", None,
        "Dairy products 0-4°C range. Power fluctuation caused complete refrigeration shutdown. Temperature rose to 6°C.",
        "medium", "dispatch_backup", "dairy_power_fluctuation",
        product_type="dairy"),

    fmt("power_outage", None,
        "Seafood -1°C to 0°C range. Power outage caused refrigeration to stop. Temperature rose to 2°C. Immediate backup required.",
        "high", "dispatch_backup", "seafood_power_outage",
        product_type="seafood"),

    fmt("power_outage", None,
        "Biological samples 2-8°C range. Complete power failure at substation caused total refrigeration loss. Temperature rose to 10°C.",
        "high", "dispatch_backup", "biological_power_outage",
        product_type="biological"),

    fmt("power_outage", None,
        "Electrical fault in wiring harness caused total power loss to cooling system. Not a compressor fault — electrical supply interrupted.",
        "high", "dispatch_backup", "wiring_harness_fault"),
]

# ═══════════════════════════════════════════════════════════════════════════
# 10 × door_left_open
# ═══════════════════════════════════════════════════════════════════════════
door_left_open = [
    fmt("door_left_open", None,
        "Rear door left open during routine inspection. Warm ambient air entering. Temperature rising slowly 0.15°C per minute over 20 minutes.",
        "medium", "notify_driver", "rear_door_left_open"),

    fmt("door_left_open", None,
        "Door left open during loading at depot. Temperature rose gradually 4°C to 6.5°C over 15 minutes. Door closed on discovery.",
        "medium", "notify_driver", "door_left_open_during_loading"),

    fmt("door_left_open", None,
        "Door seal degraded: warm air infiltrating around door frame. Temperature rising slowly 0.5°C per minute. No mechanical fault.",
        "medium", "notify_driver", "door_seal_degraded"),

    fmt("door_left_open", None,
        "Driver forgot to close door after checking load at checkpoint. Temperature rose 4°C to 6.2°C over 18 minutes.",
        "medium", "notify_driver", "door_forgotten_at_checkpoint"),

    fmt("door_left_open", None,
        "Door left ajar during unloading at customer site. Slow temperature rise 4°C to 5.5°C over 12 minutes. Physical door breach.",
        "medium", "notify_driver", "door_ajar_during_unloading"),

    fmt("door_left_open", None,
        "Door hinge damaged: door not closing fully. Warm air infiltration causing gradual temperature increase. 0.3°C per minute rise.",
        "low", "notify_driver", "door_hinge_damaged"),

    fmt("door_left_open", None,
        "Vaccine shipment 2-8°C. Temperature deviation caused by door left open during transfer at cold storage facility.",
        "high", "notify_driver", "vaccine_door_open",
        product_type="vaccines"),

    fmt("door_left_open", None,
        "Fresh produce 0-4°C range. Door opened for quality inspection and left ajar. Temperature rose to 5°C slowly over 10 minutes.",
        "medium", "notify_driver", "fresh_produce_door_open",
        product_type="fresh_produce"),

    fmt("door_left_open", None,
        "Blood products 2-6°C range. Door left open at hospital loading bay. Temperature rose to 8°C. Physical door not closed properly.",
        "high", "notify_driver", "blood_door_left_open",
        product_type="blood"),

    fmt("door_left_open", None,
        "Unloading bay incident: rear door open during vehicle-to-storage transfer for 22 minutes. Temperature rose 3.5°C to 6.8°C gradually.",
        "medium", "notify_driver", "unloading_bay_door_open"),
]

# ═══════════════════════════════════════════════════════════════════════════
# 10 × driver_error
# ═══════════════════════════════════════════════════════════════════════════
driver_error = [
    fmt("driver_error", None,
        "Driver set temperature incorrectly to 0°C instead of 4°C. Human operator mistake. Corrected after 30 minutes on alert.",
        "medium", "contact_driver", "wrong_temperature_set"),

    fmt("driver_error", None,
        "Driver ignored temperature alarm for 45 minutes. Deliberate inaction. Temperature rose to 8°C. Alarm logs confirm repeated alerts ignored.",
        "high", "contact_driver", "alarm_ignored"),

    fmt("driver_error", None,
        "Driver turned off cooling unit by mistake while adjusting cab temperature. Human error — unit switched off deliberately but incorrectly.",
        "high", "contact_driver", "cooling_unit_switched_off"),

    fmt("driver_error", None,
        "Driver misread thermostat display and set 14°C instead of 4°C. Temperature rose to 7°C before error discovered.",
        "medium", "contact_driver", "thermostat_misread"),

    fmt("driver_error", None,
        "Driver failed to verify door closure after loading. Door left partially open. Human negligence — not a mechanical fault.",
        "medium", "contact_driver", "door_not_verified"),

    fmt("driver_error", None,
        "Driver disabled temperature logging to hide deviation. Deliberate human action. Temperature records show gap in monitoring.",
        "high", "contact_driver", "logging_disabled"),

    fmt("driver_error", None,
        "Driver took unauthorised break with engine off for 40 minutes. Refrigeration powered down. Human decision caused temperature rise.",
        "high", "contact_driver", "engine_off_break"),

    fmt("driver_error", None,
        "Driver loaded cargo into wrong temperature zone. Vaccines placed in ambient section instead of refrigerated section by mistake.",
        "high", "contact_driver", "wrong_zone_loading",
        product_type="vaccines"),

    fmt("driver_error", None,
        "Driver manually overrode automatic cooling settings. Human intervention caused temperature to rise to 6.5°C before auto-recovery.",
        "medium", "contact_driver", "manual_override"),

    fmt("driver_error", None,
        "Driver communication failure: did not report observed temperature rise for 2 hours. Negligent delayed reporting by human operator.",
        "medium", "contact_driver", "delayed_reporting"),
]

# ═══════════════════════════════════════════════════════════════════════════
# 10 × extreme_weather
# ═══════════════════════════════════════════════════════════════════════════
extreme_weather = [
    fmt("extreme_weather", None,
        "Snowstorm delayed delivery and caused ambient temperature fluctuation. External weather event. Cooling system working but struggling against conditions.",
        "medium", "adjust_cooling", "snowstorm_delay"),

    fmt("extreme_weather", None,
        "Heatwave: external ambient 40°C. Cooling system working at maximum but temperature rising slowly due to external heat load. No internal fault.",
        "medium", "adjust_cooling", "heatwave_40c"),

    fmt("extreme_weather", None,
        "High ambient temperature 38°C caused internal temperature to rise from 4°C to 6.5°C. External environmental cause. Compressor intact.",
        "medium", "adjust_cooling", "high_ambient_38c"),

    fmt("extreme_weather", None,
        "Heavy rainfall caused humidity spike and condensation. External weather event. Temperature rose slightly due to latent heat from condensation.",
        "low", "adjust_humidity", "heavy_rainfall_condensation"),

    fmt("extreme_weather", None,
        "Storm caused external temperature fluctuation and wind chill interference with refrigeration panels. External weather origin.",
        "low", "adjust_cooling", "storm_interference"),

    fmt("extreme_weather", None,
        "Extreme cold external temperature -15°C: internal temperature fluctuating. Heating element working against extreme cold. External cause.",
        "low", "adjust_cooling", "extreme_cold"),

    fmt("extreme_weather", None,
        "Solar radiation on dark vehicle body caused external heat absorption. Temperature rose slowly despite working refrigeration. Environmental cause.",
        "low", "adjust_cooling", "solar_radiation"),

    fmt("extreme_weather", None,
        "Desert crossing: sustained 45°C ambient over 3 hours. Refrigeration overwhelmed by external heat load. No mechanical failure.",
        "high", "adjust_cooling", "desert_heat"),

    fmt("extreme_weather", None,
        "Tropical humidity 95%: moisture infiltration around door seals. Temperature affected by external environmental conditions.",
        "low", "adjust_humidity", "tropical_humidity"),

    fmt("extreme_weather", None,
        "Flowers 2-8°C range. Temperature rose to 10°C due to sustained heatwave. External weather cause. Cooling unit functioning normally.",
        "medium", "adjust_cooling", "flowers_heatwave",
        product_type="flowers"),
]

# ═══════════════════════════════════════════════════════════════════════════
# 10 × gps_blackout
# ═══════════════════════════════════════════════════════════════════════════
gps_blackout = [
    fmt("gps_blackout", None,
        "GPS signal lost in rural area with no cell towers. Temperature remained stable at 4°C. Navigation issue only — NO temperature anomaly.",
        "low", "manual_tracking", "gps_rural_loss"),

    fmt("gps_blackout", None,
        "GPS blackout in tunnel: signal unavailable for 8 minutes. Temperature unaffected and stable. Location tracking issue only.",
        "low", "manual_tracking", "gps_tunnel"),

    fmt("gps_blackout", None,
        "GPS lost due to solar flare interference. Temperature stable throughout. Signal-only issue. Cargo integrity maintained.",
        "low", "manual_tracking", "gps_solar_flare"),

    fmt("gps_blackout", None,
        "GPS signal lost in urban canyon between tall buildings. Temperature stable. Navigation affected only — refrigeration normal.",
        "low", "manual_tracking", "gps_urban_canyon"),

    fmt("gps_blackout", None,
        "Satellite outage caused GPS blackout across region. Temperature stable. All vehicles lost location tracking simultaneously.",
        "low", "manual_tracking", "gps_satellite_outage"),

    fmt("gps_blackout", None,
        "GPS antenna damaged by vibration. Signal intermittent. Temperature completely unaffected. Hardware communication fault only.",
        "low", "manual_tracking", "gps_antenna_damage"),

    fmt("gps_blackout", None,
        "GPS module software crash: unit rebooting. Temperature normal and stable. Tracking unavailable temporarily. Refrigeration unaffected.",
        "low", "manual_tracking", "gps_software_crash"),

    fmt("gps_blackout", None,
        "GPS jamming detected near military zone. Signal blocked deliberately. Temperature normal. Location tracking only affected.",
        "medium", "manual_tracking", "gps_jamming"),

    fmt("gps_blackout", None,
        "GPS signal lost crossing international border. Coverage gap in remote area. Temperature stable. Navigation switch to manual.",
        "low", "manual_tracking", "gps_border_coverage_gap"),

    fmt("gps_blackout", None,
        "GPS blackout during underground motorway section. 15-minute signal loss. Temperature stable. Tracking resumed on exit.",
        "low", "manual_tracking", "gps_underground_motorway"),
]

# ═══════════════════════════════════════════════════════════════════════════
# ASSEMBLE AND SAVE
# ═══════════════════════════════════════════════════════════════════════════
all_cases = (compressor + power_outage + door_left_open +
             driver_error + extreme_weather + gps_blackout)

print(f"Total cases: {len(all_cases)}")
from collections import Counter
dist = Counter(c['metadata']['type'] for c in all_cases)
print("Distribution:")
for t, n in sorted(dist.items()):
    print(f"  {t}: {n}")

# Save
out_path = Path("data/knowledge_base/anomaly_cases.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(all_cases, f, indent=2)
print(f"\n Saved to {out_path}")
print("\nNext step:")
print("  python evaluate_diagnosis_accuracy.py --rag_only --max_cases 60")
