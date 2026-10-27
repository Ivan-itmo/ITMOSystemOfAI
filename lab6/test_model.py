"""Проверки численных методов и отсутствия утечки при предобработке."""
import unittest
import numpy as np
import pandas as pd
from model import LogisticRegression, TitanicPreprocessor, sigmoid, log_loss, metrics, stratified_split


class ModelTests(unittest.TestCase):
    def test_stability(self):
        np.testing.assert_allclose(sigmoid([-1000, 0, 1000]), [0, .5, 1])
        self.assertTrue(np.isfinite(log_loss(np.array([1, 0]), np.array([-1000., 1000.]))))

    def test_gradient_and_hessian(self):
        rng = np.random.default_rng(7)
        x = np.column_stack([np.ones(20), rng.normal(size=(20, 3))])
        y = rng.integers(0, 2, 20)
        w = rng.normal(size=4)
        eps = 1e-5
        eye = np.eye(4)
        gradient = lambda a: x.T @ (sigmoid(x @ a)-y)/len(y)
        numerical = np.array([(log_loss(y, x@(w+eps*d))-log_loss(y, x@(w-eps*d)))/(2*eps) for d in eye])
        np.testing.assert_allclose(gradient(w), numerical, atol=1e-8)
        p = sigmoid(x@w)
        hessian = (x.T*(p*(1-p)))@x/len(y)
        numerical_h = np.column_stack([(gradient(w+eps*d)-gradient(w-eps*d))/(2*eps) for d in eye])
        np.testing.assert_allclose(hessian, numerical_h, atol=1e-8)

    def test_optimizers(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(300, 3))
        y = rng.binomial(1, sigmoid(x@np.array([.5, -.8, 1.])))
        gd = LogisticRegression(1., 1000).fit(x, y)
        newton = LogisticRegression(1., 10, 'newton').fit(x, y)
        np.testing.assert_allclose(gd.weights, newton.weights, atol=1e-6)
        self.assertLess(gd.loss_history[-1], gd.loss_history[0])

    def test_split(self):
        y = np.array([0]*549 + [1]*342)
        a, b = stratified_split(y)
        self.assertEqual(len(set(a) & set(b)), 0)
        self.assertEqual(len(set(a) | set(b)), len(y))
        np.testing.assert_array_equal(a, stratified_split(y)[0])
        self.assertLess(abs(y[a].mean()-y[b].mean()), .01)

    def test_preprocessing(self):
        train = pd.DataFrame({'Age': [20., 40., np.nan], 'SibSp': [0, 0, 1], 'Parch': [0, 1, 0], 'Fare': [10., 20., 30.], 'Pclass': [1, 2, 3], 'Sex': ['male', 'female', 'male'], 'Embarked': ['S', 'C', None]})
        prep = TitanicPreprocessor().fit(train)
        mean = prep.mean.copy()
        heldout = train.copy(); heldout['Age'] = [9999., np.nan, 9999.]
        self.assertTrue(np.isfinite(prep.transform(heldout)).all())
        np.testing.assert_array_equal(mean, prep.mean)
        self.assertEqual(prep.medians['Age'], 30.)
        np.testing.assert_allclose(prep.transform(train).mean(axis=0), 0, atol=1e-14)

    def test_metrics(self):
        m = metrics([1, 1, 0, 0], [1, 0, 1, 0])
        for k in ['accuracy', 'precision', 'recall', 'f1']:
            self.assertEqual(m[k], .5)
        self.assertEqual(metrics([0, 1], [0, 0])['f1'], 0.)


if __name__ == '__main__':
    unittest.main()
