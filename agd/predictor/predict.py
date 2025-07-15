import joblib

from agd.predictor.feature_extraction import smiles_to_features
from agd.utils.utils import get_project_path, pjoin


def predict(
    smiles: str,
    model_name: str,
    descriptors: list[str] = None,
    fingerprints: dict = None,
) -> float:
    """
    Predict activity/property from SMILES using a trained model and feature spec.

    Parameters:
    - smiles: str, SMILES string of the molecule
    - model_name: str, name of the model file (without .pkl)
    - descriptors: list[str], e.g., ["rdkit", "mordred"]
    - fingerprints: dict, e.g.,
        {
            "morgan": {
                "params": {
                    "radius": 2,
                    "fpSize": 1024
                }
            }
        }

    Returns:
    - float, predicted value
    """
    load_path = pjoin(get_project_path(), "resources", f"{model_name}.pkl")
    pipeline = joblib.load(load_path)

    features_df = smiles_to_features(
        smiles, descriptors=descriptors, fingerprints=fingerprints
    )

    # Align with training features
    try:
        feature_names = pipeline.named_steps["model"].feature_names_
    except AttributeError:
        feature_names = pipeline.feature_names_in_

    features_df = features_df.loc[:, feature_names]
    return pipeline.predict(features_df)[0]


def predict_rdkit_morgan(
    smiles: str, model_name: str, fpSize: int = 1024, radius: int = 2
) -> float:
    """
    Convenience wrapper for RDKit + Morgan fingerprint model prediction.

    Loads a model named <model_name>_rdkit_morgan.pkl
    """
    full_model_name = f"{model_name}_rdkit_morgan"
    return predict(
        smiles=smiles,
        model_name=full_model_name,
        descriptors=["rdkit"],
        fingerprints={"morgan": {"params": {"radius": radius, "fpSize": fpSize}}},
    )
