"""
Member 3 – Failure Category Identification and Maintenance Recommendations
(Alias module for src/maintanence_recommendations.py to match Milestone 3 specification)
"""

try:
    from .maintanence_recommendations import (
        TOOL_WEAR_CRITICAL_MIN,
        TOOL_WEAR_WARNING_MIN,
        TEMP_DIFF_MIN_K,
        SPEED_HDF_MAX_RPM,
        POWER_MIN_WATTS,
        POWER_MAX_WATTS,
        OVERSTRAIN_THRESHOLDS,
        compute_mechanical_power,
        identify_possible_failure_modes,
        generate_maintenance_recommendations,
        process_equipment_recommendations,
    )
except ImportError:
    from maintanence_recommendations import (
        TOOL_WEAR_CRITICAL_MIN,
        TOOL_WEAR_WARNING_MIN,
        TEMP_DIFF_MIN_K,
        SPEED_HDF_MAX_RPM,
        POWER_MIN_WATTS,
        POWER_MAX_WATTS,
        OVERSTRAIN_THRESHOLDS,
        compute_mechanical_power,
        identify_possible_failure_modes,
        generate_maintenance_recommendations,
        process_equipment_recommendations,
    )

__all__ = [
    "TOOL_WEAR_CRITICAL_MIN",
    "TOOL_WEAR_WARNING_MIN",
    "TEMP_DIFF_MIN_K",
    "SPEED_HDF_MAX_RPM",
    "POWER_MIN_WATTS",
    "POWER_MAX_WATTS",
    "OVERSTRAIN_THRESHOLDS",
    "compute_mechanical_power",
    "identify_possible_failure_modes",
    "generate_maintenance_recommendations",
    "process_equipment_recommendations",
]
