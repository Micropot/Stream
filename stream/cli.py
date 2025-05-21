import argparse
from stream.generate import generate_MR
import polars as pl

class Cim10Cli:
    """Class pour l'utilisation du CLI"""
    
    @staticmethod
    def parse_cli(argv):
        parser = argparse.ArgumentParser(description=__doc__)

        subparser = parser.add_subparsers(
            dest="command",
            required=True,
            help="Choose between data generation and llm fine tuning"
        )

        generate_parser = subparser.add_parser(
            "generate",
            help="Generate data"
        )
        generate_parser.add_argument(
            "--f",
            "--fictive",
            action="store_true",
            required=False,
            help="If you want to generate fictive hospital stay"
        )

        generate_parser.add_argument(
            "--r",
            "--real",
            action="store_true",
            required=False,
            help="If you want to use real hospital stay"
            
        )
        
        generate_parser.add_argument(
            "--n",
            "--number",
            required=True,
            type=int,
            help="How many CRH you want to generate"
        )


        llm_parser = subparser.add_parser(
            "llm",
            help="Train or eval model"
        )


        llm_parser.add_argument(
            "--train",
            help="argument if you want to train a model",
            action="store_true"
        )
        llm_parser.add_argument(
            "--test",
            help="load specific model and test on test set",
            action="store_true"
        )

        export_parser = subparser.add_parser(
            "export",
            help="Export trained model to ollama format"
        )
        inference_parser = subparser.add_parser(
            "inference",
            help="Use trained model to infer on a specific file"
        )
        inference_parser.add_argument(
            "--input",
            "--i",
            required=True,
            type=str,
            help="input file txt format required"
        )

        return parser.parse_args(argv)

    def run(self, argv):
        """Fonction principale du module"""
        args = self.parse_cli(argv)
        command = args.command
        if command == "generate":
            fictive = args.f
            number = args.n
            real = args.r
            if fictive:
                generate_MR(fictive=fictive, n=number)    
            if real:
                generate_MR(fictive=False, n=number)    

        if command == "llm":
            # Evite le lancement de unsloth à chaque démarrage du CLI
            from stream.finetunning import Cim10Trainer
            from stream.dataset import UnslothDatasetBuilder

            cim = Cim10Trainer()
            pl_dataset = cim.dataset
            dataset_builder = UnslothDatasetBuilder(cim.tokenizer)
            dataset = dataset_builder.build_dataset(pl_dataset)
            dataset = dataset.map(dataset_builder.formatting_prompts_func, batched=True)
            splits = dataset_builder.split_dataset(dataset)
            train = splits["train"]
            val = splits["validation"]
            test=splits["test"]
            if args.train:
                cim.set_model()
                cim.finetune_model(train=train, val=val)

            if args.test:
                cim.evaluate_model(
                                   best_model_path=cim.llm_setting.llm.best_model_path,
                                   tokenizer=dataset_builder.tokenizer,
                                   test_dataset=test,
                                   dataset_chat_template=dataset_builder.tokenize()
                           )

        if command == "export":
            from stream.export import setup_tokenizer,  export_gguf_model
            from stream.finetunning import Cim10Trainer
            from stream.dataset import UnslothDatasetBuilder
            cim = Cim10Trainer()
            dataset_builder = UnslothDatasetBuilder(cim.tokenizer)
            # tokenizer = setup_tokenizer(cim.model_name)
            export_gguf_model(model=cim.model, custom_model_path=cim.gguf_path, tokenizer=dataset_builder.tokenize(), gguf_path=cim.gguf_path)
            
            # faire ollama create MyModel -f path/to/Modelfile pour envoyer le modèle sur l'instance de server local Ollama    
        if command == "inference":
            from stream.inference import Cim10Inference
            from stream.dataset import UnslothDatasetBuilder
            cim = Cim10Inference()   
            inference_data = args.input
            with open(inference_data, "r", encoding="utf-8") as f:
                lignes = [ligne.strip() for ligne in f]

            df = pl.DataFrame({"conversations": lignes})
            text = "\n".join(df["conversations"].to_list())
            message = [
                {"role" : "user", "content": text}
            ]
            dataset_builder = UnslothDatasetBuilder(cim.tokenizer)
            cim.inference(
                        best_model_path=cim.llm_setting.llm.best_model_path,
                        tokenizer=dataset_builder.tokenizer,
                        inference_data=message,
                        dataset_chat_template=dataset_builder.tokenize()
                    )

            
