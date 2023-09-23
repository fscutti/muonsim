import ROOT as R
from muonsim import geometry as geo

# Muon energy.
energy = R.TH1F("h_energy", "Muon Energy", 1000, 0, 1000)
energy.SetTitle("Muon Energy")
energy.GetXaxis().SetTitle("Muon energy [GeV]")
energy.GetYaxis().SetTitle("Entries")

# Cosine of the zenith angle.
cos_theta = R.TH1F("h_cos_theta", "cos(theta)", 1000, 0, 1)
cos_theta.GetXaxis().SetTitle("cos(#theta)")
cos_theta.GetYaxis().SetTitle("Entries")

# True zenith angle.
true_theta = R.TH1F("h_true_theta", "true theta", 200, 0, 90)
true_theta.GetXaxis().SetTitle("#theta_{true} [deg]")
true_theta.GetYaxis().SetTitle("Entries")

# Reconstructed zenith angle.
reco_theta = R.TH1F("h_reco_theta", "reco theta", 200, 0, 90)
reco_theta.GetXaxis().SetTitle("#theta_{reco} [deg]")
reco_theta.GetYaxis().SetTitle("Entries")


# True vs reconstructed zenith angle.
true_vs_reco_theta = R.TH2F(
    "h2_true_vs_reco_theta", "reco vs true theta", 100, 0, 15, 100, 0, 15
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


# Angular distance between reconstructed and true muon.
prof2d_ang_dist = R.TProfile2D(
    "p2_ang_dist", "reco vs true angular distance", 100, 0, 15, 100, 0, 360, 0, 20
)
prof2d_ang_dist.GetXaxis().SetTitle("#theta_{true} [deg]")
prof2d_ang_dist.GetYaxis().SetTitle("#phi_{true} [deg]")
prof2d_ang_dist.GetZaxis().SetTitle("#Delta#Omega(reco - true) [deg]")

# True vs reconstructed zenith angle as a profile histogram.
prof_true_vs_reco_theta = R.TProfile(
    "p_true_vs_reco_theta", "reco vs true theta", 100, 0, 15, 0, 15
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
    "p_theta_rel_resolution", "reco theta relative resolution", 100, 0, 15, 0, 15
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
    "p_theta_resolution", "reco theta resolution", 100, 0, 15, -15, 15
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
