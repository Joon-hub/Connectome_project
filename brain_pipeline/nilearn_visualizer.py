"""
NiLearn Brain Visualization Module
==================================
Advanced brain visualization for connectivity classification results using NiLearn.
Supports Schaefer cortical atlas and Tian subcortical atlas.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import nibabel as nib
from typing import Dict, Optional
from nilearn import plotting, datasets, surface

warnings.filterwarnings('ignore', category=FutureWarning)


class NiLearnVisualizer:
    """Create publication-ready brain visualizations using NiLearn."""

    def __init__(self, output_dir: str = "outputs", config=None):
        """Initialize visualizer and load brain atlases."""
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)  # ensure folder exists

        self.schaefer_200_17 = None
        self.schaefer_labels = None
        self.tian_labels = None
        self._load_atlases()

    # --------------------------------------------------------------------------
    # LOAD ATLASES
    # --------------------------------------------------------------------------
    def _load_atlases(self):
        """Load Schaefer cortical atlas and Tian subcortical placeholders."""
        print("Loading brain atlases...")

        try:
            self.schaefer_200_17 = datasets.fetch_atlas_schaefer_2018(
                n_rois=200, yeo_networks=17, resolution_mm=1, verbose=0
            )
            print("✓ Loaded Schaefer 200x17 cortical atlas")

            self.schaefer_labels = [
                label.decode("utf-8") if isinstance(label, bytes) else label
                for label in self.schaefer_200_17["labels"]
            ]
        except Exception as e:
            print(f"⚠ Could not load Schaefer atlas: {e}")

        print("\nNote: Tian subcortical atlas requires manual download from:")
        print("https://www.nitrc.org/projects/tian2020_mni")
        print("Using placeholder for subcortical regions\n")

        # placeholder subcortical labels (for visualization completeness)
        self.tian_labels = [
            'lh-THA', 'rh-THA', 'lh-CAU', 'rh-CAU', 'lh-PUT', 'rh-PUT',
            'lh-GP', 'rh-GP', 'lh-HIP', 'rh-HIP', 'lh-AMY', 'rh-AMY',
            'lh-NAc', 'rh-NAc'
        ] + [f'Subcortical_{i}' for i in range(18)]

    # --------------------------------------------------------------------------
    # MAIN: CREATE ERROR MAP
    # --------------------------------------------------------------------------
    def create_error_brain_map(self, error_map_df: pd.DataFrame,
                               output_name: str = "brain_error_map") -> Dict:
        """
        Create a NIfTI brain map and visualizations from misclassification rates.

        Args:
            error_map_df: DataFrame with columns ['region_name', 'misclassification_rate']
            output_name: Base filename for saved outputs

        Returns:
            Dict of saved output file paths
        """
        outputs = {}
        region_errors = self._map_errors_to_atlas(error_map_df)
        error_img = self._create_error_nifti(region_errors)

        if error_img is not None:
            nifti_path = self.output_dir / f"{output_name}.nii.gz"
            nib.save(error_img, nifti_path)
            outputs["nifti"] = str(nifti_path)
            print(f"✓ Saved error map NIfTI: {nifti_path}")

            outputs.update(self._create_all_visualizations(error_img, output_name))
        else:
            print("⚠ Failed to create NIfTI image")

        return outputs

    # --------------------------------------------------------------------------
    # MAP ERRORS TO ATLAS
    # --------------------------------------------------------------------------
    def _map_errors_to_atlas(self, error_map_df: pd.DataFrame) -> np.ndarray:
        """Map each region's misclassification rate to an atlas index."""
        n_regions = 232  # 200 Schaefer + 32 Tian
        region_errors = np.zeros(n_regions)

        for _, row in error_map_df.iterrows():
            region_name = row["region_name"]
            error_rate = row["misclassification_rate"]
            atlas_idx = self._find_region_index(region_name)
            if atlas_idx is not None:
                region_errors[atlas_idx] = error_rate

        return region_errors

    def _find_region_index(self, region_name: str) -> Optional[int]:
        """Find index of a region name in the atlas labels."""
        if self.schaefer_labels:
            for i, label in enumerate(self.schaefer_labels):
                if region_name in label or label in region_name:
                    return i

        subcortical_offset = 200
        if any(sub in region_name for sub in ['THA', 'CAU', 'PUT', 'GP', 'HIP', 'AMY', 'NAc']):
            for i, tian_label in enumerate(self.tian_labels):
                if any(part in region_name for part in tian_label.split('-')):
                    return subcortical_offset + i
        return None

    # --------------------------------------------------------------------------
    # CREATE NIFTI IMAGE
    # --------------------------------------------------------------------------
    def _create_error_nifti(self, region_errors: np.ndarray) -> Optional[nib.Nifti1Image]:
        """Construct NIfTI from region-wise error values."""
        if self.schaefer_200_17 is None:
            print("⚠ Atlas not loaded, cannot create NIfTI")
            return None

        atlas_img = nib.load(self.schaefer_200_17["maps"])
        atlas_data = atlas_img.get_fdata()
        error_data = np.zeros_like(atlas_data)

        for region_idx in range(200):
            if region_errors[region_idx] > 0:
                mask = atlas_data == (region_idx + 1)
                error_data[mask] = region_errors[region_idx]

        return nib.Nifti1Image(error_data, atlas_img.affine, atlas_img.header)

    # --------------------------------------------------------------------------
    # CREATE VISUALIZATIONS
    # --------------------------------------------------------------------------
    def _create_all_visualizations(self, error_img, output_name: str) -> Dict[str, str]:
        """Generate all visualization types."""
        outputs = {}
        glass = self.output_dir / f"{output_name}_glass.png"
        surf = self.output_dir / f"{output_name}_surface.png"
        slices = self.output_dir / f"{output_name}_slices.png"
        html = self.output_dir / f"{output_name}_interactive.html"

        self._create_glass_brain(error_img, glass)
        self._create_surface_plot(error_img, surf)
        self._create_slice_mosaic(error_img, slices)
        self._create_interactive_html(error_img, html)

        outputs.update({
            "glass_brain": str(glass),
            "surface": str(surf),
            "slices": str(slices),
            "html": str(html)
        })
        return outputs

    def _create_glass_brain(self, img, out_path: Path):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, mode in zip(axes, ["x", "y", "z"]):
            plotting.plot_glass_brain(
                img, display_mode=mode, colorbar=False, cmap="hot",
                vmax=0.5, axes=ax, title=f"{mode.upper()} view"
            )
        plt.suptitle("Brain Connectivity Error Map - Glass Brain", fontsize=14)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Created glass brain: {out_path}")

    def _create_surface_plot(self, img, out_path: Path):
        """Create cortical surface visualization."""
        fsavg = datasets.fetch_surf_fsaverage(mesh="fsaverage5")

        # Use pial surface for projection
        texture_left = surface.vol_to_surf(img, fsavg["pial_left"])
        texture_right = surface.vol_to_surf(img, fsavg["pial_right"])

        fig, axes = plt.subplots(2, 4, figsize=(20, 10), subplot_kw={"projection": "3d"})
        views = ["lateral", "medial", "dorsal", "ventral"]

        for i, view in enumerate(views):
            plotting.plot_surf_stat_map(
                fsavg["pial_left"], texture_left, hemi="left", view=view,
                colorbar=False, cmap="hot", vmax=0.5, axes=axes[0, i]
            )
            plotting.plot_surf_stat_map(
                fsavg["pial_right"], texture_right, hemi="right", view=view,
                colorbar=False, cmap="hot", vmax=0.5, axes=axes[1, i]
            )

        plt.suptitle("Cortical Surface Error Map", fontsize=14)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Created surface plot: {out_path}")


    def _create_slice_mosaic(self, img, out_path: Path):
        fig = plt.figure(figsize=(15, 12))
        for i, (mode, title) in enumerate([("z", "Axial"), ("y", "Coronal"), ("x", "Sagittal")]):
            ax = plt.subplot(3, 1, i + 1)
            plotting.plot_stat_map(
                img, display_mode=mode, cut_coords=8,
                colorbar=(i == 2), cmap="hot", vmax=0.5,
                axes=ax, title=f"{title} Slices"
            )
        plt.suptitle("Brain Error Map - Slice Mosaic", fontsize=14)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Created slice mosaic: {out_path}")

    def _create_interactive_html(self, img, out_path: Path):
        """Create interactive HTML surface visualization."""
        fsavg = datasets.fetch_surf_fsaverage(mesh="fsaverage5")

        # Use pial surface instead of inflated (for compatibility)
        texture_left = surface.vol_to_surf(img, fsavg["pial_left"])
        view = plotting.view_surf(
            surf_mesh=fsavg["pial_left"],
            surf_map=texture_left,
            cmap="hot",
            symmetric_cmap=False,
            vmax=0.5,
            title="Interactive Brain Error Map"
        )

        # Save HTML
        view.save_as_html(out_path)
        print(f"✓ Created interactive HTML: {out_path}")
        
    def create_network_comparison(self, error_map_rest: pd.DataFrame,
                              error_map_task: pd.DataFrame,
                              output_name: str = "network_comparison") -> str:
        """
        Create a side-by-side comparison of rest vs task misclassification error maps.

        Args:
            error_map_rest: DataFrame with resting-state errors
            error_map_task: DataFrame with task-state errors
            output_name: Base name for the output file

        Returns:
            Path to the saved comparison figure
        """
        # Merge rest and task errors on region name
        merged = pd.merge(
            error_map_rest[["region_name", "misclassification_rate"]],
            error_map_task[["region_name", "misclassification_rate"]],
            on="region_name",
            suffixes=("_rest", "_task")
        )

        merged["error_change"] = (
            merged["misclassification_rate_task"] -
            merged["misclassification_rate_rest"]
        )

        # Map to atlas indices
        rest_errors = self._map_errors_to_atlas(error_map_rest)
        task_errors = self._map_errors_to_atlas(error_map_task)
        change_errors = self._map_errors_to_atlas(
            merged[["region_name", "error_change"]].rename(columns={"error_change": "misclassification_rate"})
        )

        # Convert to NIfTI images
        rest_img = self._create_error_nifti(rest_errors)
        task_img = self._create_error_nifti(task_errors)
        change_img = self._create_error_nifti(change_errors)

        # Create visualization figure
        fig = plt.figure(figsize=(20, 8))

        # Rest condition
        ax1 = plt.subplot(1, 3, 1)
        plotting.plot_glass_brain(
            rest_img, display_mode="lyrz", colorbar=True,
            cmap="Blues", vmax=0.5, axes=ax1, title="Resting State"
        )

        # Task condition
        ax2 = plt.subplot(1, 3, 2)
        plotting.plot_glass_brain(
            task_img, display_mode="lyrz", colorbar=True,
            cmap="Reds", vmax=0.5, axes=ax2, title="Task State"
        )

        # Change map (Task - Rest)
        ax3 = plt.subplot(1, 3, 3)
        plotting.plot_glass_brain(
            change_img, display_mode="lyrz", colorbar=True,
            cmap="RdBu_r", vmin=-0.3, vmax=0.3,
            axes=ax3, title="Task - Rest (Change)"
        )

        plt.suptitle("Connectivity Classification: Rest vs Task Comparison", fontsize=16, fontweight="bold")

        output_path = self.output_dir / f"{output_name}.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Created comparison figure: {output_path}")

        return str(output_path)

    

