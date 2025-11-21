"""
Measure Crosswalk: Maps quality measures across CMS programs

This module identifies which measures appear in multiple programs, enabling
cross-program impact analysis and ROI prioritization.

Programs covered:
- Star Ratings (SR)
- Hospital Value-Based Purchasing (HVBP)
- Hospital Readmissions Reduction Program (HRRP)
- Hospital-Acquired Condition Reduction Program (HACRP)
"""

# ==============================================================================
# MORTALITY MEASURES - Appear in Star Ratings + HVBP
# ==============================================================================

MORTALITY_MEASURES = {
    'MORT_30_AMI': {
        'name': 'Heart Attack (AMI) 30-Day Mortality',
        'programs': ['STAR_RATINGS', 'HVBP'],
        'star_rating_group': 'Mortality',
        'hvbp_domain': 'Clinical Outcomes',
        'measure_id_star': 'MORT_30_AMI',
        'measure_id_hvbp': 'MORT_30_AMI'
    },
    'MORT_30_HF': {
        'name': 'Heart Failure 30-Day Mortality',
        'programs': ['STAR_RATINGS', 'HVBP'],
        'star_rating_group': 'Mortality',
        'hvbp_domain': 'Clinical Outcomes',
        'measure_id_star': 'MORT_30_HF',
        'measure_id_hvbp': 'MORT_30_HF'
    },
    'MORT_30_PN': {
        'name': 'Pneumonia 30-Day Mortality',
        'programs': ['STAR_RATINGS', 'HVBP'],
        'star_rating_group': 'Mortality',
        'hvbp_domain': 'Clinical Outcomes',
        'measure_id_star': 'MORT_30_PN',
        'measure_id_hvbp': 'MORT_30_PN'
    },
    'MORT_30_COPD': {
        'name': 'COPD 30-Day Mortality',
        'programs': ['STAR_RATINGS'],
        'star_rating_group': 'Mortality',
        'measure_id_star': 'MORT_30_COPD'
    },
    'MORT_30_CABG': {
        'name': 'CABG 30-Day Mortality',
        'programs': ['STAR_RATINGS', 'HVBP'],
        'star_rating_group': 'Mortality',
        'hvbp_domain': 'Clinical Outcomes',
        'measure_id_star': 'MORT_30_CABG',
        'measure_id_hvbp': 'MORT_30_CABG'
    },
    'MORT_30_STK': {
        'name': 'Stroke 30-Day Mortality',
        'programs': ['STAR_RATINGS'],
        'star_rating_group': 'Mortality',
        'measure_id_star': 'MORT_30_STK'
    }
}

# ==============================================================================
# READMISSION MEASURES - Appear in Star Ratings + HRRP
# ==============================================================================

READMISSION_MEASURES = {
    'READM_30_AMI': {
        'name': 'Heart Attack (AMI) 30-Day Readmission',
        'programs': ['STAR_RATINGS', 'HRRP'],
        'star_rating_group': 'Readmission',
        'measure_id_star': 'READM_30_AMI',
        'measure_id_hrrp': 'AMI'
    },
    'READM_30_HF': {
        'name': 'Heart Failure 30-Day Readmission',
        'programs': ['STAR_RATINGS', 'HRRP'],
        'star_rating_group': 'Readmission',
        'measure_id_star': 'READM_30_HF',
        'measure_id_hrrp': 'HF'
    },
    'READM_30_PN': {
        'name': 'Pneumonia 30-Day Readmission',
        'programs': ['STAR_RATINGS', 'HRRP'],
        'star_rating_group': 'Readmission',
        'measure_id_star': 'READM_30_PN',
        'measure_id_hrrp': 'PN'
    },
    'READM_30_COPD': {
        'name': 'COPD 30-Day Readmission',
        'programs': ['STAR_RATINGS', 'HRRP'],
        'star_rating_group': 'Readmission',
        'measure_id_star': 'READM_30_COPD',
        'measure_id_hrrp': 'COPD'
    },
    'READM_30_CABG': {
        'name': 'CABG 30-Day Readmission',
        'programs': ['STAR_RATINGS', 'HRRP'],
        'star_rating_group': 'Readmission',
        'measure_id_star': 'READM_30_CABG',
        'measure_id_hrrp': 'CABG'
    },
    'READM_30_HIP_KNEE': {
        'name': 'Hip/Knee Replacement 30-Day Readmission',
        'programs': ['STAR_RATINGS', 'HRRP'],
        'star_rating_group': 'Readmission',
        'measure_id_star': 'READM_30_HIP_KNEE',
        'measure_id_hrrp': 'THA_TKA'
    },
    'READM_30_HOSP_WIDE': {
        'name': 'Hospital-Wide All-Cause Readmission',
        'programs': ['STAR_RATINGS'],
        'star_rating_group': 'Readmission',
        'measure_id_star': 'READM_30_HOSP_WIDE'
    }
}

# ==============================================================================
# SAFETY MEASURES (HAIs) - Appear in Star Ratings + HVBP + HACRP
# ==============================================================================

SAFETY_MEASURES = {
    'HAI_1': {
        'name': 'Central Line-Associated Bloodstream Infection (CLABSI)',
        'programs': ['STAR_RATINGS', 'HVBP', 'HACRP'],
        'star_rating_group': 'Safety of Care',
        'hvbp_domain': 'Safety',
        'hacrp_measure': 'CLABSI',
        'measure_id_star': 'HAI_1',
        'measure_id_hvbp': 'HAI_1_SIR',
        'measure_id_hacrp': 'CLABSI_SIR'
    },
    'HAI_2': {
        'name': 'Catheter-Associated Urinary Tract Infection (CAUTI)',
        'programs': ['STAR_RATINGS', 'HVBP', 'HACRP'],
        'star_rating_group': 'Safety of Care',
        'hvbp_domain': 'Safety',
        'hacrp_measure': 'CAUTI',
        'measure_id_star': 'HAI_2',
        'measure_id_hvbp': 'HAI_2_SIR',
        'measure_id_hacrp': 'CAUTI_SIR'
    },
    'HAI_3': {
        'name': 'Surgical Site Infection - Colon Surgery',
        'programs': ['STAR_RATINGS', 'HVBP', 'HACRP'],
        'star_rating_group': 'Safety of Care',
        'hvbp_domain': 'Safety',
        'hacrp_measure': 'SSI_COLON',
        'measure_id_star': 'HAI_3',
        'measure_id_hvbp': 'HAI_3_SIR',
        'measure_id_hacrp': 'SSI_COLON_SIR'
    },
    'HAI_4': {
        'name': 'Surgical Site Infection - Abdominal Hysterectomy',
        'programs': ['STAR_RATINGS', 'HVBP', 'HACRP'],
        'star_rating_group': 'Safety of Care',
        'hvbp_domain': 'Safety',
        'hacrp_measure': 'SSI_HYST',
        'measure_id_star': 'HAI_4',
        'measure_id_hvbp': 'HAI_4_SIR',
        'measure_id_hacrp': 'SSI_HYST_SIR'
    },
    'HAI_5': {
        'name': 'MRSA Bacteremia',
        'programs': ['STAR_RATINGS', 'HVBP', 'HACRP'],
        'star_rating_group': 'Safety of Care',
        'hvbp_domain': 'Safety',
        'hacrp_measure': 'MRSA',
        'measure_id_star': 'HAI_5',
        'measure_id_hvbp': 'HAI_5_SIR',
        'measure_id_hacrp': 'MRSA_SIR'
    },
    'HAI_6': {
        'name': 'Clostridioides difficile Infection (CDI)',
        'programs': ['STAR_RATINGS', 'HVBP', 'HACRP'],
        'star_rating_group': 'Safety of Care',
        'hvbp_domain': 'Safety',
        'hacrp_measure': 'CDI',
        'measure_id_star': 'HAI_6',
        'measure_id_hvbp': 'HAI_6_SIR',
        'measure_id_hacrp': 'CDI_SIR'
    },
    'PSI_90': {
        'name': 'Patient Safety Indicator 90 (Composite)',
        'programs': ['STAR_RATINGS', 'HACRP'],
        'star_rating_group': 'Safety of Care',
        'hacrp_measure': 'PSI_90',
        'measure_id_star': 'PSI_90_SAFETY',
        'measure_id_hacrp': 'PSI_90'
    }
}

# ==============================================================================
# PATIENT EXPERIENCE (HCAHPS) - Appear in Star Ratings + HVBP
# ==============================================================================

PATIENT_EXPERIENCE_MEASURES = {
    'H_COMP_1': {
        'name': 'Communication with Nurses',
        'programs': ['STAR_RATINGS', 'HVBP'],
        'star_rating_group': 'Patient Experience',
        'hvbp_domain': 'Person and Community Engagement',
        'measure_id_star': 'H_COMP_1',
        'measure_id_hvbp': 'HCAHPS_COMP_1'
    },
    'H_COMP_2': {
        'name': 'Communication with Doctors',
        'programs': ['STAR_RATINGS', 'HVBP'],
        'star_rating_group': 'Patient Experience',
        'hvbp_domain': 'Person and Community Engagement',
        'measure_id_star': 'H_COMP_2',
        'measure_id_hvbp': 'HCAHPS_COMP_2'
    },
    'H_COMP_3': {
        'name': 'Staff Responsiveness',
        'programs': ['STAR_RATINGS', 'HVBP'],
        'star_rating_group': 'Patient Experience',
        'hvbp_domain': 'Person and Community Engagement',
        'measure_id_star': 'H_COMP_3',
        'measure_id_hvbp': 'HCAHPS_COMP_3'
    },
    'H_COMP_5': {
        'name': 'Communication About Medicines',
        'programs': ['STAR_RATINGS', 'HVBP'],
        'star_rating_group': 'Patient Experience',
        'hvbp_domain': 'Person and Community Engagement',
        'measure_id_star': 'H_COMP_5',
        'measure_id_hvbp': 'HCAHPS_COMP_5'
    },
    'H_COMP_6': {
        'name': 'Discharge Information',
        'programs': ['STAR_RATINGS', 'HVBP'],
        'star_rating_group': 'Patient Experience',
        'hvbp_domain': 'Person and Community Engagement',
        'measure_id_star': 'H_COMP_6',
        'measure_id_hvbp': 'HCAHPS_COMP_6'
    },
    'H_COMP_7': {
        'name': 'Care Transition',
        'programs': ['STAR_RATINGS', 'HVBP'],
        'star_rating_group': 'Patient Experience',
        'hvbp_domain': 'Person and Community Engagement',
        'measure_id_star': 'H_COMP_7',
        'measure_id_hvbp': 'HCAHPS_COMP_7'
    },
    'H_CLEAN': {
        'name': 'Hospital Cleanliness',
        'programs': ['STAR_RATINGS', 'HVBP'],
        'star_rating_group': 'Patient Experience',
        'hvbp_domain': 'Person and Community Engagement',
        'measure_id_star': 'H_CLEAN',
        'measure_id_hvbp': 'HCAHPS_CLEAN'
    },
    'H_QUIET': {
        'name': 'Hospital Quietness',
        'programs': ['STAR_RATINGS', 'HVBP'],
        'star_rating_group': 'Patient Experience',
        'hvbp_domain': 'Person and Community Engagement',
        'measure_id_star': 'H_QUIET',
        'measure_id_hvbp': 'HCAHPS_QUIET'
    },
    'H_HSP_RATING': {
        'name': 'Hospital Overall Rating',
        'programs': ['STAR_RATINGS', 'HVBP'],
        'star_rating_group': 'Patient Experience',
        'hvbp_domain': 'Person and Community Engagement',
        'measure_id_star': 'H_HSP_RATING',
        'measure_id_hvbp': 'HCAHPS_RATING'
    }
}

# ==============================================================================
# COMPLICATIONS - Appear in Star Ratings + HVBP
# ==============================================================================

COMPLICATION_MEASURES = {
    'COMP_HIP_KNEE': {
        'name': 'Hip/Knee Complication Rate',
        'programs': ['STAR_RATINGS', 'HVBP'],
        'star_rating_group': 'Safety of Care',
        'hvbp_domain': 'Clinical Outcomes',
        'measure_id_star': 'COMP_HIP_KNEE',
        'measure_id_hvbp': 'THA_TKA_COMP'
    }
}

# ==============================================================================
# COMBINED CROSSWALK - All measures
# ==============================================================================

ALL_MEASURES = {
    **MORTALITY_MEASURES,
    **READMISSION_MEASURES,
    **SAFETY_MEASURES,
    **PATIENT_EXPERIENCE_MEASURES,
    **COMPLICATION_MEASURES
}

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_measures_by_program(program_name):
    """Get all measures for a specific program"""
    return {
        measure_id: info
        for measure_id, info in ALL_MEASURES.items()
        if program_name in info['programs']
    }

def get_cross_program_measures():
    """Get measures that appear in multiple programs"""
    return {
        measure_id: info
        for measure_id, info in ALL_MEASURES.items()
        if len(info['programs']) > 1
    }

def get_measure_programs(measure_id):
    """Get list of programs affected by a specific measure"""
    if measure_id in ALL_MEASURES:
        return ALL_MEASURES[measure_id]['programs']
    return []

def get_cascading_impact_measures():
    """Get measures with the highest cascading impact (3 programs)"""
    return {
        measure_id: info
        for measure_id, info in ALL_MEASURES.items()
        if len(info['programs']) >= 3
    }

def get_program_impact_summary():
    """Generate summary of cross-program impacts"""
    summary = {
        'total_measures': len(ALL_MEASURES),
        'cross_program_measures': len(get_cross_program_measures()),
        'triple_impact_measures': len(get_cascading_impact_measures()),
        'by_program_count': {
            'STAR_RATINGS': len(get_measures_by_program('STAR_RATINGS')),
            'HVBP': len(get_measures_by_program('HVBP')),
            'HRRP': len(get_measures_by_program('HRRP')),
            'HACRP': len(get_measures_by_program('HACRP'))
        }
    }
    return summary

# ==============================================================================
# PROGRAM WEIGHTS FOR IMPACT SCORING
# ==============================================================================

PROGRAM_WEIGHTS = {
    'STAR_RATINGS': {
        'financial_weight': 0.3,  # Indirect financial impact (market share, reputation)
        'strategic_weight': 0.4   # Strategic importance (public perception)
    },
    'HVBP': {
        'financial_weight': 0.25,  # 2% of Medicare payments at stake
        'strategic_weight': 0.2
    },
    'HRRP': {
        'financial_weight': 0.30,  # Up to 3% penalty
        'strategic_weight': 0.2
    },
    'HACRP': {
        'financial_weight': 0.15,  # 1% penalty for worst quartile
        'strategic_weight': 0.2
    }
}
