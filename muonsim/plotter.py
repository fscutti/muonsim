import os
import ROOT as R


def save_canvases(input_file, output_directory, output_file=None, save_eps=False):
    """This function grabs objects from a ROOT file and prints
    a canvas for each of them."""

    input_name = os.path.split(input_file.GetName())[-1]

    # This is currently buggy
    if output_file is None:
        output_file = f"plots_{input_name}"

    output_file = R.TFile.Open(os.path.join(output_directory, output_file), "RECREATE")

    objects = input_file.GetListOfKeys()

    canvases = {}

    for _obj in objects:
        obj_name = _obj.GetName()

        obj = input_file.Get(obj_name)

        canvases[obj_name] = R.TCanvas(f"c_{obj_name}", f"{obj_name}", 900, 800)
        canvases[obj_name].SetTickx()
        canvases[obj_name].SetTicky()

        canvases[obj_name].SetGridx()
        canvases[obj_name].SetGridy()

        canvases[obj_name].cd()

        obj.SetLineWidth(3)
        obj.SetLineColor(R.kRed)
        obj.Draw()

        # canvases[obj_name].SetOptStat(0)

        output_file.WriteObject(canvases[obj_name], canvases[obj_name].GetName())

        if save_eps:
            out_canvas = os.path.join(
                output_directory, canvases[obj_name].GetName() + ".eps"
            )
            canvases[obj_name].SaveAs(out_canvas)


# EOF
