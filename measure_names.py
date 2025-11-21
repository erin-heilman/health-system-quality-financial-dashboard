"""
Human-readable names for CMS quality measures
"""

MEASURE_NAMES = {
    # Mortality Measures
    'MORT_30_AMI': 'Heart Attack 30-Day Mortality',
    'MORT_30_HF': 'Heart Failure 30-Day Mortality',
    'MORT_30_PN': 'Pneumonia 30-Day Mortality',
    'MORT_30_STK': 'Stroke 30-Day Mortality',
    'MORT_30_CABG': 'CABG 30-Day Mortality',
    'MORT_30_COPD': 'COPD 30-Day Mortality',
    'PSI_4_SURG_COMP': 'Surgical Complications Death Rate',

    # Safety of Care
    'HAI_1': 'Central Line Bloodstream Infection (CLABSI)',
    'HAI_2': 'Catheter-Associated UTI (CAUTI)',
    'HAI_3': 'Surgical Site Infection - Colon',
    'HAI_4': 'Surgical Site Infection - Hysterectomy',
    'HAI_5': 'MRSA Bloodstream Infection',
    'HAI_6': 'C. difficile Infection',
    'COMP_HIP_KNEE': 'Hip/Knee Complication Rate',
    'PSI_90_SAFETY': 'Patient Safety Composite',

    # Readmission
    'READM_30_AMI': 'Heart Attack 30-Day Readmission',
    'READM_30_HF': 'Heart Failure 30-Day Readmission',
    'READM_30_PN': 'Pneumonia 30-Day Readmission',
    'READM_30_COPD': 'COPD 30-Day Readmission',
    'READM_30_CABG': 'CABG 30-Day Readmission',
    'READM_30_HOSP_WIDE': 'Hospital-Wide Readmission',
    'OP_32': 'Facility 7-Day Visit Rate',
    'OP_35': 'ED Visit Rate - AMI',
    'OP_36': 'ED Visit Rate - Heart Failure',

    # Patient Experience
    'H_COMP_1': 'Communication with Nurses',
    'H_COMP_2': 'Communication with Doctors',
    'H_COMP_3': 'Staff Responsiveness',
    'H_COMP_5': 'Communication About Medicines',
    'H_COMP_6': 'Discharge Information',
    'H_COMP_7': 'Care Transition',
    'H_CLEAN': 'Hospital Cleanliness',
    'H_QUIET': 'Hospital Quietness',

    # Timely & Effective Care
    'SEP_1': 'Sepsis Care',
    'IMM_3': 'Healthcare Workers Flu Vaccination',
    'PC_01': 'Elective Delivery Before 39 Weeks',
    'OP_22': 'ED Left Without Being Seen',
    'OP_18B': 'ED Median Time to Pain Med',
    'OP_8': 'MRI Lumbar Spine Imaging',
    'OP_10': 'Abdomen CT Dual Contrast',
    'OP_13': 'Cardiac Imaging Stress Test',
    'OP_29': 'Brain CT/MRI After Concussion',
    'OP_33': 'External Beam Radiotherapy',
    'SAFE_USE_OF_OPIOIDS': 'Safe Use of Opioids',
    'ED_2B': 'ED Admit Decision to Departure',
}

# Group weights and measure counts
GROUP_WEIGHTS = {
    'Mortality': 0.22,
    'Safety of Care': 0.22,
    'Readmission': 0.22,
    'Patient Experience': 0.22,
    'Timely & Effective Care': 0.12
}

GROUP_MEASURE_COUNTS = {
    'Mortality': 7,
    'Safety of Care': 8,
    'Readmission': 11,
    'Patient Experience': 8,
    'Timely & Effective Care': 12
}
