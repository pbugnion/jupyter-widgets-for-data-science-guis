
import ipywidgets as widgets

from . import Classifier, Transaction


class Labeller(object):

    def __init__(self, transactions, categories):
        self.transactions = transactions
        self.categories = categories
        self.classifier = Classifier()
        self._train()
        self.container = widgets.VBox([])
        self.transaction_container = widgets.VBox(
            [],
            layout={'height': '15em', 'overflow_y': 'scroll'}
        )
        self.retrain_button = None

    def _train(self):
        training_set = [
            transaction for transaction in self.transactions
            if transaction.category
        ]
        self.classifier.train(training_set, self.categories)

    def _render_row(self, transaction, prediction):
        [predicted_category, probability] = prediction
        return widgets.HBox([
            widgets.Label(transaction.memo, layout={'width': '250px'}),
            widgets.Dropdown(
                options=[''] + self.categories,
                value=transaction.category,
                layout={'width': '100px'}
            ),
            widgets.Label(predicted_category, layout={'width': '100px'}),
            widgets.FloatProgress(
                value=probability,
                min=1./3.,
                max=1.0,
                layout={'width': '150px'}
            ),
        ], layout={'flex': '1 0 auto'})

    def _get_updated_transactions(self):
        transactions = []
        for row_widget in self.transaction_container.children:
            transaction = Transaction(
                row_widget.children[0].value,
                row_widget.children[1].value
            )
            transactions.append(transaction)
        return transactions

    def _render_headers(self):
        return widgets.HBox([
            widgets.HTML('<b>Memo</b>', layout={'width': '250px'}),
            widgets.HTML('<b>Correct</b>', layout={'width': '100px'}),
            widgets.HTML('<b>Predicted</b>', layout={'width': '100px'}),
            widgets.HTML('<b>Probability</b>', layout={'width': '150px'})
        ])

    def _render_controls(self):
        self.retrain_button = widgets.Button(description='retrain')
        self.retrain_button.on_click(lambda evt: self.retrain())
        return widgets.HBox([self.retrain_button])

    def retrain(self):
        # disable retrain button while retraining
        if self.retrain_button is not None:
            self.retrain_button.description = 'retraining...'
            self.retrain_button.disabled = True
        self.transactions = self._get_updated_transactions()
        self._train()
        self.render()
        if self.retrain_button is not None:
            self.retrain_button.description = 'retrain'
            self.retrain_button.disabled = False

    def render(self):
        predictions = self.classifier.predict_proba(self.transactions)
        row_widgets = []
        zipped_transaction_predictions = zip(self.transactions, predictions)
        sorted_transaction_predictions = sorted(
            zipped_transaction_predictions, key=lambda entry: entry[1][1])
        for (transaction, prediction) in sorted_transaction_predictions:
            row_widget = self._render_row(transaction, prediction)
            row_widgets.append(row_widget)
        headers = self._render_headers()
        controls = self._render_controls()
        self.transaction_container.children = row_widgets
        self.container.children = [
            headers, self.transaction_container, controls
        ]
        return self.container

