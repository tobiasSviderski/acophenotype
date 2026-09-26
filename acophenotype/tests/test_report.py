"""The HTML report renders correctly from a profile dict (no torch/audio needed)."""
from __future__ import annotations

from acophenotype import build_html, AcousticProfile


def test_report_is_self_contained_html(profile_data):
    html = build_html(profile_data)
    assert html.startswith("<!DOCTYPE html>")
    assert html.rstrip().endswith("</html>")
    # no external assets: no http(s) links, no <script src>, no <link rel=stylesheet>
    assert "http://" not in html and "https://" not in html
    assert "<script" not in html
    assert 'rel="stylesheet"' not in html


def test_report_contains_key_sections(profile_data):
    html = build_html(profile_data)
    assert "Fusion result" in html
    assert "Acoustic biomarkers" in html
    assert "Emotion dynamics" in html
    assert "Research use only" in html
    # verdict + gauge value + a per-stream label
    assert "consistent with ad" in html.lower()
    assert "0.78" in html
    assert "biomarkers" in html


def test_report_flags_are_rendered(profile_data):
    html = build_html(profile_data)
    assert "Outside typical range" in html
    assert "phon jitter local" in html  # prettified flag name


def test_biomarker_only_profile_hides_fusion(profile_data):
    profile_data["fusion"] = None
    profile_data["biomarkers"]["reference_available"] = False
    profile_data["biomarkers"]["percentiles"] = {}
    profile_data["biomarkers"]["bands"] = {}
    profile_data["umap"] = None
    html = build_html(profile_data)
    assert "Fusion result" not in html
    assert "Biomarker profile (no fused prediction)" in html
    # raw biomarker values still render, just without percentile context
    assert "Acoustic biomarkers" in html
    assert "temp syllable rate" in html
    assert "cohort interquartile range" not in html
    # still valid, self-contained
    assert html.startswith("<!DOCTYPE html>")


def test_emotion_trajectory_is_plotted(profile_data):
    html = build_html(profile_data)
    assert "Trajectory over time" in html
    # one polyline per emotion series, drawn as SVG (not an image)
    assert html.count("<polyline") == 3
    # no NaN/inf leaked into the plotted coordinates ("provenance" contains "nan", so
    # check the polyline point data specifically rather than the whole document)
    import re
    for pts in re.findall(r'<polyline points="([^"]*)"', html):
        assert "nan" not in pts.lower() and "inf" not in pts.lower()
    # variability stats table, not just means
    assert "Per-emotion statistics" in html
    for stat in ("mean", "std", "range", "cv"):
        assert f">{stat}<" in html
    assert "angry" in html  # all emotions, not a hardcoded few


def test_emotion_section_without_trajectory_degrades(profile_data):
    profile_data["emotion"].pop("trajectory")
    html = build_html(profile_data)
    assert "no per-chunk trajectory" in html.lower()
    assert "Per-emotion statistics" in html  # table still renders


def test_umap_section(profile_data):
    html = build_html(profile_data)
    assert "Position among the reference cohort" in html
    assert "<circle" in html          # the reference clouds
    assert "this speaker" in html


def test_umap_omitted_when_absent(profile_data):
    profile_data["umap"] = None
    html = build_html(profile_data)
    assert "Position among the reference cohort" not in html


def test_biomarkers_split_into_compared_and_uncompared(profile_data):
    html = build_html(profile_data)
    assert "Compared with the reference cohort" in html
    assert "Measured without a reference" in html
    # features without percentiles go to the plain table, with their family
    table = html.split("Measured without a reference")[1]
    for feat, fam in [("comp dfa", "Complexity"), ("rhythm npvi", "Rhythm")]:
        assert feat in table and fam in table
    assert "1,800" in html            # spec_centroid formatted, not 1.8e+03


def test_no_empty_comparison_bars(profile_data):
    """A feature without a reference must not get a comparison bar."""
    html = build_html(profile_data)
    compared = html.split("Compared with the reference cohort")[1].split(
        "Measured without a reference")[0]
    for feat in ("comp dfa", "rhythm npvi", "dyn teager energy mean"):
        assert feat not in compared
    assert '<div class="bar"></div>' not in html   # the old empty-bar placeholder


def test_none_band_does_not_crash(profile_data):
    # spec_centroid has an explicit None band in the fixture
    html = build_html(profile_data)
    assert "spec centroid" in html


def test_ad_consistency_annotation(profile_data):
    html = build_html(profile_data)
    # jitter at the 95th percentile, direction "increases" -> AD-consistent
    assert "AD-consistent" in html


def test_provenance_block_has_reproducibility_details(profile_data):
    html = build_html(profile_data)
    assert "Provenance &amp; reproducibility" in html
    assert "2026-07-20" in html                     # timestamp
    assert "scikit-learn=1.7.2" in html             # pickle-sensitive pins
    assert "voxmarkers_feature_schema=1.0" in html  # frozen schema version
    assert "binary_biomarkers_LGBMClassifier" in html
    assert "Limitations" in html
    assert "not a medical device" in html.lower()


def test_profile_from_dict_to_html(profile_data, tmp_path):
    prof = AcousticProfile.from_dict(profile_data)
    assert prof.prediction == "AD"
    assert abs(prof.probability - 0.78) < 1e-9
    assert prof.per_stream["biomarkers"] == 0.41
    out = tmp_path / "r.html"
    prof.to_html(out)
    assert out.exists() and out.stat().st_size > 1000
