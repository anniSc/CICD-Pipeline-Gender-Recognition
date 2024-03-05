import unittest
from unittest.mock import MagicMock, patch
from model_test import ModelTester
import torch

class TestModelTester(unittest.TestCase):
    def test_test_model_robustness(self):
        # Erstelle ein Mock-Modell
        model = MagicMock()
        # Erstelle einen Mock-Test-Dataloader
        test_dataloader = MagicMock()
        # Erstelle ein Mock-Gerät
        device = MagicMock()
        
        # Rufe die Methode test_model_robustness auf
        result = ModelTester.test_model_robustness(model, test_dataloader, device)
        
        # Überprüfe, ob das Ergebnis ein Tupel ist
        self.assertIsInstance(result, tuple)
        # Überprüfe, ob das Ergebnistupel 4 Elemente hat
        self.assertEqual(len(result), 4)
        # Überprüfe, ob jedes Element im Ergebnistupel ein Float ist
        for element in result:
            self.assertIsInstance(element, float)
    
    def test_get_model_path(self):
        # Erstelle ein Mock-Verzeichnis
        directory = MagicMock()
        
        # Rufe die Methode get_model_path auf
        result = ModelTester.get_model_path(directory)
        
        # Überprüfe, ob das Ergebnis ein String oder None ist
        self.assertIsInstance(result, (str, type(None)))
    
    def test_add_noise(self):
        # Erstelle einen Mock-Bilder-Tensor
        images = MagicMock()
        # Erstelle einen Mock-Rauschfaktor
        noise_factor = MagicMock()
        
        # Rufe die Methode add_noise auf
        result = ModelTester.add_noise(images, noise_factor)
        
        # Überprüfe, ob das Ergebnis ein Tensor ist
        self.assertIsInstance(result, torch.Tensor)
    
    def test_add_noise_and_test_robustness(self):
        # Erstelle ein Mock-Modell
        model = MagicMock()
        # Erstelle einen Mock-Test-Dataloader
        test_dataloader = MagicMock()
        # Erstelle ein Mock-Gerät
        device = MagicMock()
        # Erstelle einen Mock-Rauschfaktor
        noise_factor = MagicMock()
        
        # Rufe die Methode add_noise_and_test_robustness auf
        ModelTester.add_noise_and_test_robustness(model, test_dataloader, device, noise_factor)
        
        # Überprüfe, ob die Methode test_model_robustness aufgerufen wurde
        model.test_model_robustness.assert_called_with(model, test_dataloader, device)
    
    def test_evaluate_model(self):
        # Erstelle ein Mock-Modell
        model = MagicMock()
        # Erstelle einen Mock-Test-Dataloader
        test_dataloader = MagicMock()
        
        # Rufe die Methode evaluate_model auf
        result = ModelTester.evaluate_model(model, test_dataloader)
        
        # Überprüfe, ob das Ergebnis ein Tupel ist
        self.assertIsInstance(result, tuple)
        # Überprüfe, ob das Ergebnistupel 4 Elemente hat
        self.assertEqual(len(result), 4)
        # Überprüfe, ob jedes Element im Ergebnistupel ein Float ist
        for element in result:
            self.assertIsInstance(element, float)
    
    def test_test_predicts(self):
        # Erstelle ein Mock-Modell
        model = MagicMock()
        # Erstelle einen Mock-Test-Dataloader
        test_dataloader = MagicMock()
        # Erstelle ein Mock-Gerät
        device = MagicMock()
        
        # Rufe die Methode test_predicts auf
        ModelTester.test_predicts(model, test_dataloader, device)
        
        # Überprüfe, ob die Modellmethodenaufrufe gemacht wurden
        model.assert_called_with(model, test_dataloader, device)
    
    def test_test_noise_robustness(self):
        # Erstelle ein Mock-Modell
        model = MagicMock()
        # Erstelle einen Mock-Test-Dataloader
        test_dataloader = MagicMock()
        # Erstelle ein Mock-Gerät
        device = MagicMock()
        # Erstelle Mock-Rauschparameter
        start_noise = MagicMock()
        end_noise = MagicMock()
        step = MagicMock()
        savefig_path = MagicMock()
        
        # Rufe die Methode test_noise_robustness auf
        ModelTester.test_noise_robustness(model, test_dataloader, device, start_noise, end_noise, step, savefig_path)
        
        # Überprüfe, ob die Methode test_model_robustness mehrmals aufgerufen wurde
        model.test_model_robustness.assert_called()
        self.assertEqual(model.test_model_robustness.call_count, 11)

if __name__ == '__main__':
    unittest.main()