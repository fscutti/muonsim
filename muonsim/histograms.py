import ROOT as R
from muonsim import geometry as geo

# Muon energy.
energy = R.TH1F("h_energy", "Muon Energy", 1000, 0, 1000)
energy.SetTitle("Generated muon Energy")
energy.GetXaxis().SetTitle("Muon energy [GeV]")
energy.GetYaxis().SetTitle("Entries")

# Cosine of the zenith angle.
cos_theta = R.TH1F("h_cos_theta", "generated cos(theta)", 1000, 0, 1)
cos_theta.GetXaxis().SetTitle("cos(#theta)")
cos_theta.GetYaxis().SetTitle("Entries")

# True zenith angle.
true_theta = R.TH1F("h_true_theta", "generated true theta", 200, 0, 90)
true_theta.GetXaxis().SetTitle("#theta_{true} [deg]")
true_theta.GetYaxis().SetTitle("Entries")

# Reconstructed zenith angle.
reco_theta = R.TH1F("h_reco_theta", "reco theta", 200, 0, 90)
reco_theta.GetXaxis().SetTitle("#theta_{reco} [deg]")
reco_theta.GetYaxis().SetTitle("Entries")


# Reconstructed muons distribution.
reco_raw = R.TH2F(
    "h2_reco_raw", "reconstructed muons, non-weighted", 90, 0, 90, 360, 0, 360
)
reco_raw.GetXaxis().SetTitle("#theta_{reco} [deg]")
reco_raw.GetYaxis().SetTitle("#phi_{reco} [deg]")
reco_raw.GetZaxis().SetTitle("N_{#mu}")

reco_weighted = R.TH2F(
    # "h2_reco_weighted", "reconstructed muons, weighted", 6, 0, 90, 15, 0, 360
    "h2_reco_weighted",
    "reconstructed muons, weighted",
    500,
    0,
    90,
    100,
    0,
    360,
)
reco_weighted.GetXaxis().SetTitle("#theta_{reco} [deg]")
reco_weighted.GetYaxis().SetTitle("#phi_{reco} [deg]")
reco_weighted.GetZaxis().SetTitle("N_{#mu}")

# True muons distribution.
true_raw = R.TH2F("h2_true_raw", "true muons, non-weighted", 500, 0, 90, 100, 0, 360)
true_raw.GetXaxis().SetTitle("#theta_{true} [deg]")
true_raw.GetYaxis().SetTitle("#phi_{true} [deg]")
true_raw.GetZaxis().SetTitle("N_{#mu}")

true_weighted = R.TH2F(
    # "h2_true_weighted", "true muons, weighted", 6, 0, 90, 16, 0, 360
    "h2_true_weighted",
    "true muons, weighted",
    30,
    0,
    90,
    20,
    0,
    360,
)
true_weighted.GetXaxis().SetTitle("#theta_{true} [deg]")
true_weighted.GetYaxis().SetTitle("#phi_{true} [deg]")
true_weighted.GetZaxis().SetTitle("N_{#mu}")


# Correction for reconstructed muons.
reco_correction = R.TH2F(
    "h2_reco_correction", "reconstruction weights", 500, 0, 90, 100, 0, 360
)
reco_correction.GetXaxis().SetTitle("#theta_{reco} [deg]")
reco_correction.GetYaxis().SetTitle("#phi_{reco} [deg]")
reco_correction.GetZaxis().SetTitle("[a.u.]")


# Validation of reconstructed muons.
reco_validation = R.TH2F(
    "h2_reco_validation",
    "reconstruction validation",
    30,
    0,
    90,
    20,
    0,
    360
    # "h2_reco_validation", "reconstruction validation", 500, 0, 90, 100, 0, 360
)
reco_validation.GetXaxis().SetTitle("#theta_{reco} [deg]")
reco_validation.GetYaxis().SetTitle("#phi_{reco} [deg]")
reco_validation.GetZaxis().SetTitle("[a.u.]")


# True vs reconstructed zenith angle.
true_vs_reco_theta = R.TH2F(
    "h2_true_vs_reco_theta", "reco vs true theta", 500, 0, 50, 500, 0, 50
)
true_vs_reco_theta.GetXaxis().SetTitle("#theta_{true} [deg]")
true_vs_reco_theta.GetYaxis().SetTitle("#theta_{reco} [deg]")
true_vs_reco_theta.GetZaxis().SetTitle("Entries")

# True vs reconstructed azimuth angle.
true_vs_reco_phi = R.TH2F(
    "h2_true_vs_reco_phi", "reco vs true phi", 100, 0, 360, 100, 0, 360
)
true_vs_reco_phi.GetXaxis().SetTitle("#phi_{true} [deg]")
true_vs_reco_phi.GetYaxis().SetTitle("#phi_{reco} [deg]")
true_vs_reco_phi.GetZaxis().SetTitle("Entries")

# Angular acceptance vs true angles.
prof2d_ang_acc_vs_true_high_reso = R.TProfile2D(
    "p2_ang_acc_vs_true_high_reso",
    "angular acceptance vs true angles",
    500,
    0,
    90,
    100,
    0,
    360,
    0,
    2,
)
prof2d_ang_acc_vs_true_high_reso.GetXaxis().SetTitle("#theta_{true} [deg]")
prof2d_ang_acc_vs_true_high_reso.GetYaxis().SetTitle("#phi_{true} [deg]")
prof2d_ang_acc_vs_true_high_reso.GetZaxis().SetTitle("N(Reco) / N(Total)")

prof2d_ang_acc_vs_true_low_reso = R.TProfile2D(
    "p2_ang_acc_vs_true_low_reso",
    "angular acceptance vs true angles",
    30,
    0,
    90,
    8,
    0,
    360,
    0,
    2,
)
prof2d_ang_acc_vs_true_low_reso.GetXaxis().SetTitle("#theta_{true} [deg]")
prof2d_ang_acc_vs_true_low_reso.GetYaxis().SetTitle("#phi_{true} [deg]")
prof2d_ang_acc_vs_true_low_reso.GetZaxis().SetTitle("N(Reco) / N(Total)")

# Angular acceptance vs reconstructed angles.
prof2d_ang_acc_vs_reco = R.TProfile2D(
    "p2_ang_acc_vs_reco",
    "angular acceptance vs reconstructed angles",
    500,
    0,
    50,
    100,
    0,
    360,
    0,
    2,
)
prof2d_ang_acc_vs_reco.GetXaxis().SetTitle("#theta_{reco} [deg]")
prof2d_ang_acc_vs_reco.GetYaxis().SetTitle("#phi_{reco} [deg]")
prof2d_ang_acc_vs_reco.GetZaxis().SetTitle("N(Reco) / N(Total)")

# Angular distance between reconstructed and true muon as a function of the true angles.
prof2d_ang_dist_vs_true = R.TProfile2D(
    "p2_ang_dist_vs_true",
    "reco - true angular distance vs true angles",
    500,
    0,
    50,
    100,
    0,
    360,
    0,
    20,
)
prof2d_ang_dist_vs_true.GetXaxis().SetTitle("#theta_{true} [deg]")
prof2d_ang_dist_vs_true.GetYaxis().SetTitle("#phi_{true} [deg]")
prof2d_ang_dist_vs_true.GetZaxis().SetTitle("#Delta#Omega(reco - true) [deg]")

# Angular distance between reconstructed and true muon as a function of the reconstructed angles.
prof2d_ang_dist_vs_reco = R.TProfile2D(
    "p2_ang_dist_vs_reco",
    "reco - true angular distance vs reconstructed angles",
    500,
    0,
    50,
    100,
    0,
    360,
    0,
    20,
)
prof2d_ang_dist_vs_reco.GetXaxis().SetTitle("#theta_{reco} [deg]")
prof2d_ang_dist_vs_reco.GetYaxis().SetTitle("#phi_{reco} [deg]")
prof2d_ang_dist_vs_reco.GetZaxis().SetTitle("#Delta#Omega(reco - true) [deg]")


# Elevation RMS between reconstructed and true muon as a function of the reconstructed angles.
prof2d_el_rms_vs_reco = R.TProfile2D(
    "p2_el_rms_vs_reco",
    "Elevation 2 #times RMS vs reconstructed angles",
    30,
    0,
    90,
    180,
    0,
    360,
    0,
    90,
)
prof2d_el_rms_vs_reco.GetXaxis().SetTitle("#eta_{reco} [deg]")
prof2d_el_rms_vs_reco.GetYaxis().SetTitle("#phi_{reco} [deg]")
prof2d_el_rms_vs_reco.GetZaxis().SetTitle("2 RMS (#eta_{reco} - #eta_{true}) [deg]")


# Azimuth RMS between reconstructed and true muon as a function of the reconstructed angles.
prof2d_az_rms_vs_reco = R.TProfile2D(
    "p2_az_rms_vs_reco",
    "Azimuth 2 #times RMS vs reconstructed angles",
    30,
    0,
    90,
    180,
    0,
    360,
    0,
    360,
)
prof2d_az_rms_vs_reco.GetXaxis().SetTitle("#eta_{reco} [deg]")
prof2d_az_rms_vs_reco.GetYaxis().SetTitle("#phi_{reco} [deg]")
prof2d_az_rms_vs_reco.GetZaxis().SetTitle("2 RMS (#phi_{reco} - #phi_{true}) [deg^{2}]")


# Elevation RMS between reconstructed and true muon as a function of the true angles.
prof2d_el_rms_vs_true = R.TProfile2D(
    "p2_el_rms_vs_true",
    "Elevation 2 #times RMS vs true angles",
    30,
    0,
    90,
    180,
    0,
    360,
    0,
    90,
)
prof2d_el_rms_vs_true.GetXaxis().SetTitle("#eta_{true} [deg]")
prof2d_el_rms_vs_true.GetYaxis().SetTitle("#phi_{true} [deg]")
prof2d_el_rms_vs_true.GetZaxis().SetTitle("2 RMS (#eta_{reco} - #eta_{true}) [deg]")


# Azimuth RMS between reconstructed and true muon as a function of the true angles.
prof2d_az_rms_vs_true = R.TProfile2D(
    "p2_az_rms_vs_true",
    "Azimuth 2 #times RMS vs true angles",
    30,
    0,
    90,
    180,
    0,
    360,
    0,
    360,
)
prof2d_az_rms_vs_true.GetXaxis().SetTitle("#eta_{reco} [deg]")
prof2d_az_rms_vs_true.GetYaxis().SetTitle("#phi_{reco} [deg]")
prof2d_az_rms_vs_true.GetZaxis().SetTitle("2 RMS (#phi_{reco} - #phi_{true}) [deg]")


# Elevation RMS between reconstructed and true muon as a function of the reconstructed angles.
el_max_dist_vs_reco = R.TH2F(
    "h2_max_el_dist_vs_reco",
    "Max Elevation vs reconstructed angles",
    30,
    0,
    90,
    180,
    0,
    360,
)
el_max_dist_vs_reco.GetXaxis().SetTitle("#eta_{reco} [deg]")
el_max_dist_vs_reco.GetYaxis().SetTitle("#phi_{reco} [deg]")
el_max_dist_vs_reco.GetZaxis().SetTitle("Max |#eta_{reco} - #eta_{true}| [deg]")


# Azimuth RMS between reconstructed and true muon as a function of the reconstructed angles.
az_max_dist_vs_reco = R.TH2F(
    "h2_max_az_dist_vs_reco",
    "Max Azimuth vs reconstructed angles",
    30,
    0,
    90,
    180,
    0,
    360,
)
az_max_dist_vs_reco.GetXaxis().SetTitle("#eta_{reco} [deg]")
az_max_dist_vs_reco.GetYaxis().SetTitle("#phi_{reco} [deg]")
az_max_dist_vs_reco.GetZaxis().SetTitle("Max |#phi_{reco} - #phi_{true}| [deg^{2}]")


# Elevation RMS between reconstructed and true muon as a function of the true angles.
el_max_dist_vs_true = R.TH2F(
    "h2_max_el_dist_vs_true",
    "Max Elevation vs true angles",
    30,
    0,
    90,
    180,
    0,
    360,
)
el_max_dist_vs_true.GetXaxis().SetTitle("#eta_{true} [deg]")
el_max_dist_vs_true.GetYaxis().SetTitle("#phi_{true} [deg]")
el_max_dist_vs_true.GetZaxis().SetTitle("Max |#eta_{reco} - #eta_{true}| [deg]")


# Azimuth RMS between reconstructed and true muon as a function of the true angles.
az_max_dist_vs_true = R.TH2F(
    "h2_max_az_dist_vs_true",
    "Max Azimuth dist vs true angles",
    30,
    0,
    90,
    180,
    0,
    360,
)
az_max_dist_vs_true.GetXaxis().SetTitle("#eta_{true} [deg]")
az_max_dist_vs_true.GetYaxis().SetTitle("#phi_{true} [deg]")
az_max_dist_vs_true.GetZaxis().SetTitle("Max |#phi_{reco} - #phi_{true}| [deg]")


# True vs reconstructed zenith angle as a profile histogram.
prof_true_vs_reco_theta = R.TProfile(
    "p_true_vs_reco_theta", "reco vs true theta", 500, 0, 50, 0, 50
)
prof_true_vs_reco_theta.GetXaxis().SetTitle("#theta_{true} [deg]")
prof_true_vs_reco_theta.GetYaxis().SetTitle("#theta_{reco} [deg]")

# True vs reconstructed azimuth angle as a profile histogram.
prof_true_vs_reco_phi = R.TProfile(
    "p_true_vs_reco_phi", "reco vs true phi", 100, 0, 360, 0, 360
)
prof_true_vs_reco_phi.GetXaxis().SetTitle("#phi_{true} [deg]")
prof_true_vs_reco_phi.GetYaxis().SetTitle("#phi_{reco} [deg]")

# Theta relative resolution.
prof_theta_rel_resolution = R.TProfile(
    "p_theta_rel_resolution", "reco theta relative resolution", 500, 0, 50, 0, 50
)
prof_theta_rel_resolution.GetXaxis().SetTitle("#theta_{true} [deg]")
prof_theta_rel_resolution.GetYaxis().SetTitle(
    "|#theta_{true} - #theta_{reco}| / #theta_{true}"
)


# Phi relative resolution.
prof_phi_rel_resolution = R.TProfile(
    "p_phi_rel_resolution", "reco phi relative resolution", 100, 0, 360, 0, 360
)
prof_phi_rel_resolution.GetXaxis().SetTitle("#phi_{true} [deg]")
prof_phi_rel_resolution.GetYaxis().SetTitle("|#phi_{true} - #phi_{reco}| / #phi_{true}")

# Theta resolution.
prof_theta_resolution = R.TProfile(
    "p_theta_resolution", "reco theta resolution", 500, 0, 50, -50, 50
)
prof_theta_resolution.GetXaxis().SetTitle("#theta_{true} [deg]")
prof_theta_resolution.GetYaxis().SetTitle("#theta_{true} - #theta_{reco} [deg]")

# Phi resolution.
prof_phi_resolution = R.TProfile(
    "p_phi_resolution", "reco phi resolution", 100, 0, 360, -360, 360
)
prof_phi_resolution.GetXaxis().SetTitle("#phi_{true} [deg]")
prof_phi_resolution.GetYaxis().SetTitle("#phi_{true} - #phi_{reco} [deg]")


# Total path length.
total_path_length = R.TH1F("h_total_path_length", "Total path length", 100, 0, 10)
total_path_length.GetXaxis().SetTitle("Path length [cm]")
total_path_length.GetYaxis().SetTitle("Entries")

# Top panel path length.
top_path_length = R.TH1F("h_top_path_length", "Top panel path length", 100, 0, 10)
top_path_length.GetXaxis().SetTitle("Path length [cm]")
top_path_length.GetYaxis().SetTitle("Entries")

# Bottom panel path length.
bottom_path_length = R.TH1F(
    "h_bottom_path_length", "Bottom panel path length", 100, 0, 10
)
bottom_path_length.GetXaxis().SetTitle("Path length [cm]")
bottom_path_length.GetYaxis().SetTitle("Entries")


_path_length = {}


def path_length(detector, element):
    """Function to dynamically generate histograms."""
    if element in _path_length:
        return _path_length[element]

    else:
        _path_length[element] = R.TH1F(
            f"h_{element}_path_length", f"h_{element}_path_length", 100, 0, 10
        )
        _path_length[element].GetXaxis().SetTitle("Path length [cm]")
        _path_length[element].GetYaxis().SetTitle("Entries")

    return _path_length[element]


# EOF
