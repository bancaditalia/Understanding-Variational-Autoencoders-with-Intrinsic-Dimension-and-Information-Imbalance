import os
import argparse
from vae_class import VAE_Model


def main(args):
    
    for dim in args.latent_dim:
        model = VAE_Model(model_kwargs={"latent_dim": dim})
        loss = model.train(dataset_name=args.dataset, num_epochs=0, batch_size=args.batch_size)
        model.save_network(model_name=f"epoch_0_dim_{dim}")
        for epoch in range(1, args.epochs + 1):
            loss = model.train(dataset_name=args.dataset, num_epochs=1)
            if epoch in args.save_interval:
                model.save_network(model_name=f"epoch_{epoch}_dim_{dim}")
                print(f"Model saved at epoch {epoch}")
            loss_save_path = os.path.join('./data/vae_losses', f"loss_epoch_{epoch}_dim_{dim}.txt")
            try:
                with open(loss_save_path, 'w') as f:
                    print(f"Writing loss {loss} to {loss_save_path}")
                    f.write(f"{loss}\n")
            except Exception as e:
                print(f"Error writing file {loss_save_path}: {e}")
                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model on a specified dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["cifar10", "mnist"],
        help="Dataset to train on (cifar10, mnist).",
    )
    parser.add_argument(
        "--epochs", type=int, required=True, help="Number of epochs to train the model."
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        nargs="+",
        default=[1, 2, 5, 10, 20, 50, 75, 100, 150, 200],
        help="Number of epochs between saving the model.",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16, 32, 64, 128],
        help="Dimension of the latent space.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for training."
    )
    args = parser.parse_args()
    main(args)

