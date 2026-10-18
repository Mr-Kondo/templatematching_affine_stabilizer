#!/usr/bin/env python3
"""Compatibility entry: python 02_code/affine_template_matching.py [options]."""
from stabilizer_core import (
    output_files,
    open_video,
    verify_video,
    run_pipeline,
    _quad,
    refine,
    crop,
    crop_lrbt,
    template_integer,
    validate_template,
    validate_templates,
    load_templates,
    TemplateMatcher,
    AffineCorrector,
    MATCH_SCORE_MIN,
    MATCH_CLEAR_MIN,
    MATCH_DELTA_MAX,
    MATCH_EXPAND_FACTOR,
    MATCH_CCOEFF_ALARM,
    MATCH_RESEARCH_GAIN,
    TEMPLATE_FIELDS,
    EST_INLIER_COUNT_MIN,
    EST_INLIER_RATIO_MIN,
    EST_RESIDUAL_MAX,
    EST_SPREAD_MIN,
    EST_DELTA_T_MAX,
    EST_DELTA_SCALE_MAX,
    EST_DELTA_ROT_MAX,
)
from stabilizer_gui import TemplateGUI
from stabilizer_cli import main

if __name__ == '__main__':
    main()
