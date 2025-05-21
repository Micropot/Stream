from pydantic import BaseModel
import yaml

class OllamaConfig(BaseModel):
    host: str
    model: str
    
class HfConfig(BaseModel):
    model: str
    
class LlmConfig(BaseModel):
    epoch: int
    batch_size: int
    lr: float    
    best_model_path: str
    r: int
    lora_alpha: int
    
class PathConfig(BaseModel):
    train_dataset: str
    pmsi_dataset: str
    gguf_path: str

class Config(BaseModel): # config locale
    ollama: OllamaConfig
    
class TrainConfig(BaseModel):
    llm: LlmConfig
    hf: HfConfig
    path: PathConfig
    
def load_config(yaml_file: str, config_class: BaseModel) -> BaseModel:
    """
    Charge et valide une configuration depuis un fichier YAML en utilisant Pydantic.

    Args:
        yaml_file (str): Chemin du fichier YAML.
        config_class (BaseModel): Classe Pydantic pour valider la configuration.

    Returns:
        BaseModel: Une instance de la classe Pydantic contenant la configuration validée.
    """
    try:
        with open(yaml_file, "r") as file:
            data = yaml.safe_load(file)
            # print(data)
            # print("Data saved correctly")
        return config_class(**data)
    except FileNotFoundError:
        raise FileNotFoundError(f"Le fichier {yaml_file} est introuvable.")
    except yaml.YAMLError as e:
        raise ValueError(f"Erreur lors de la lecture du fichier YAML : {e}")
    except Exception as e:
        raise ValueError(
            f"Erreur lors de la validation de la configuration : {e}")
