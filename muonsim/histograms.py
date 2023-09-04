import ROOT as R
from muonsim import geometry as geo

# Muon energy.
energy = R.TH1F("h_energy", "Muon Energy", 1000, 0, 1000)
energy.SetTitle("Muon Energy")
energy.GetXaxis().SetTitle("Muon energy [GeV]")
energy.GetYaxis().SetTitle("Entries")

# Cosine of the zenith angle.
cos_theta = R.TH1F("h_cos_theta", "cos(theta)", 100, 0, 1)
cos_theta.GetXaxis().SetTitle("cos(theta)")
cos_theta.GetYaxis().SetTitle("Entries")

# Path length in detector submodule.
path_length = {}
for element in geo.block_detector:
    path_length[element] = R.TH1F(
        f"h_{element}_path_length", f"h_{element}_path_length", 1000, 0, 10
    )
    path_length[element].GetXaxis().SetTitle("path length [cm]")
    path_length[element].GetYaxis().SetTitle("Entries")

# Total path length.
total_path_length = R.TH1F("h_total_path_length", "Total path length", 1000, 0, 10)
total_path_length.GetXaxis().SetTitle("path length [cm]")
total_path_length.GetYaxis().SetTitle("Entries")


top_path_length = R.TH1F("h_top_path_length", "Top panel path length", 1000, 0, 10)
top_path_length.GetXaxis().SetTitle("path length [cm]")
top_path_length.GetYaxis().SetTitle("Entries")

bottom_path_length = R.TH1F(
    "h_bottom_path_length", "Bottom panel path length", 1000, 0, 10
)
bottom_path_length.GetXaxis().SetTitle("path length [cm]")
bottom_path_length.GetYaxis().SetTitle("Entries")

# EOF
