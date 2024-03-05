import unittest
from unittest.mock import MagicMock, patch

from scipy import datasets
from model_train import DataLoaderModelTrain, Main, SimpleCNN, SimpleCNN2, SimpleCNN3, Trainer
import os
from torchvision import datasets
# class TestModelTrain(unittest.TestCase):
#     """
#     Eine Testklasse zum Testen der ModelTrain-Klasse.

#     Diese Klasse enthält Testmethoden für verschiedene Modelle und den Trainer.
#     """

#     def __init__(self, methodName: str = "runTest") -> None:
#         super().__init__(methodName)
#         self.model = SimpleCNN()
#         self.model2 = SimpleCNN2()
#         self.model3 = SimpleCNN3()

#     def test_SimpleCNN(self):
#         model = SimpleCNN()
#         self.assertEqual(model.name, "SimpleCNN")

#     def test_SimpleCNN2(self):
#         model = SimpleCNN2()
#         self.assertEqual(model.name, "SimpleCNN2")

#     def test_SimpleCNN3(self):
#         model = SimpleCNN3()
#         self.assertEqual(model.name, "SimpleCNN3")

#     def test_Trainer(self):
#         train_dataloader = ...
#         test_dataloader = ...
#         epochs = 10
#         batch_size = 32
#         criterion = ...
#         optimizer = ...
#         trainer = Trainer(self.model, train_dataloader, test_dataloader, epochs, optimizer, criterion, batch_size)
#         self.assertEqual(trainer.epochs, epochs)
#         self.assertEqual(trainer.batch_size, batch_size)

#     def test_Trainer2(self):
#         train_dataloader = ...
#         test_dataloader = ...
#         epochs = 10
#         batch_size = 32
#         criterion = ...
#         optimizer = ...
#         trainer = Trainer(self.model2, train_dataloader, test_dataloader, epochs, optimizer, criterion, batch_size)
#         self.assertEqual(trainer.epochs, epochs)
#         self.assertEqual(trainer.batch_size, batch_size)

#     def test_Trainer3(self):
#             """
#             Testet die Trainer-Klasse mit den gegebenen Parametern.

#             Parameters:
#                 - train_dataloader: Der Trainings-Dataloader.
#                 - test_dataloader: Der Test-Dataloader.
#                 - epochs: Die Anzahl der Epochen.
#                 - batch_size: Die Batch-Größe.
#                 - criterion: Das Verlustfunktionskriterium.
#                 - optimizer: Der Optimierungsalgorithmus.
            
#             Returns:
#                 None
#             """
#             train_dataloader = ...
#             test_dataloader = ...
#             epochs = 10
#             batch_size = 32
#             criterion = ...
#             optimizer = ...
#             trainer = Trainer(self.model3, train_dataloader, test_dataloader, epochs, optimizer, criterion, batch_size)
#             self.assertEqual(trainer.epochs, epochs)
#             self.assertEqual(trainer.batch_size, batch_size)


# if __name__ == '__main__':
#     unittest.main()

class TestModelTrain(unittest.TestCase):
    """
    Eine Testklasse zum Testen der ModelTrain-Klasse.

    Diese Klasse enthält Testmethoden für verschiedene Modelle und den Trainer.
    """

    def __init__(self, methodName: str = "runTest") -> None:
        """
        Initialisiert eine Instanz der TestModelTrain-Klasse.

        :param methodName: Der Name der Testmethode.
        """
        super().__init__(methodName)
        self.model = SimpleCNN()
        self.model2 = SimpleCNN2()
        self.model3 = SimpleCNN3()

    def test_SimpleCNN(self):
        model = SimpleCNN()
        self.assertEqual(model.name, "SimpleCNN")

    def test_SimpleCNN2(self):
        model = SimpleCNN2()
        self.assertEqual(model.name, "SimpleCNN2")

    def test_SimpleCNN3(self):
        model = SimpleCNN3()
        self.assertEqual(model.name, "SimpleCNN3")

    def test_Trainer(self):
        train_dataloader = ...
        test_dataloader = ...
        epochs = 10
        batch_size = 32
        criterion = ...
        optimizer = ...
        trainer = Trainer(self.model, train_dataloader, test_dataloader, epochs, optimizer, criterion, batch_size)
        self.assertEqual(trainer.epochs, epochs)
        self.assertEqual(trainer.batch_size, batch_size)

    def test_Trainer2(self):
        train_dataloader = ...
        test_dataloader = ...
        epochs = 10
        batch_size = 32
        criterion = ...
        optimizer = ...
        trainer = Trainer(self.model2, train_dataloader, test_dataloader, epochs, optimizer, criterion, batch_size)
        self.assertEqual(trainer.epochs, epochs)
        self.assertEqual(trainer.batch_size, batch_size)

    def test_Trainer3(self):
        train_dataloader = ...
        test_dataloader = ...
        epochs = 10
        batch_size = 32
        criterion = ...
        optimizer = ...
        trainer = Trainer(self.model3, train_dataloader, test_dataloader, epochs, optimizer, criterion, batch_size)
        self.assertEqual(trainer.epochs, epochs)
        self.assertEqual(trainer.batch_size, batch_size)

    def test_Main(self):
        batch_size = 64
        epochs = 50
        test_dir = "data/train-test-data/test"
        transform = None
        train_dir = "data/train-test-data/train"
        train_dataloader = ...
        test_dataloader = ...
        model = SimpleCNN()
        model_save_path = "model/PyTorch_Trained_Models/"
        model_test_path = "test/model_to_be_tested/"

        main = Main(
            batch_size=batch_size,
            epochs=epochs,
            test_dir=test_dir,
            transform=transform,
            train_dir=train_dir,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            model=model,
            model_save_path=model_save_path,
            model_test_path=model_test_path
        )

        self.assertEqual(main.batch_size, batch_size)
        self.assertEqual(main.epochs, epochs)
        self.assertEqual(main.test_dir, test_dir)
        self.assertEqual(main.transform, transform)
        self.assertEqual(main.train_dir, train_dir)
        self.assertEqual(main.train_dataloader, train_dataloader)
        self.assertEqual(main.test_dataloader, test_dataloader)
        self.assertEqual(main.model, model)
        self.assertEqual(main.model_save_path, model_save_path)
        self.assertEqual(main.model_test_path, model_test_path)

    def test_clean_up_pth(self):
        directory = "test/model_to_be_tested/"
        if os.path.exists(directory):
            self.assertEqual(Main.clean_up_pth(directory), "Modell gelöscht!")
        else:
            self.assertEqual(Main.clean_up_pth(directory), "default.txt")
    @patch('matplotlib.pyplot')
    def test_plot_cpu_memory_usage(self, mock_plt):
        cpu_percentages = [10, 20, 30]
        memory_percentages = [40, 50, 60]
        time_stamps = [1, 2, 3]

        Trainer.plot_cpu_memory_usage(self,cpu_percentages=cpu_percentages, memory_percentages=memory_percentages,time_stamps= time_stamps)

        # Überprüfen, ob die Plot-Funktionen aufgerufen wurden
        # self.assertTrue(mock_plt.plot.called)
    #     self.assertTrue(mock_plt.savefig.called)
    

    # @patch.object(datasets.ImageFolder, '__init__', return_value=None)
    # @patch('torch.utils.data.DataLoader')
    # def test_load_data(self, mock_dataloader, mock_dataset):
    #     test_dir = Main.test_dir
    #     train_dir = Main.train_dir
    #     transform = MagicMock()
    #     batch_size = 32
    #     mock_dataloader.return_value = MagicMock()
    #     train_dataloader, test_dataloader = DataLoaderModelTrain.load_data(test_dir, train_dir, transform, batch_size)
    #     Überprüfen, ob die DataLoader korrekt erstellt wurden
    #     self.assertTrue(mock_dataloader.called)
    #     self.assertEqual(mock_dataloader.call_count, 2)
        
        
if __name__ == '__main__':
    unittest.main()