import { Component } from '@angular/core';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrl: './app.component.css',
})
export class App {
  protected readonly models = [
    { name: 'Logistic Regression', accuracy: '81.59%', auc: '0.894', tag: 'Interpretable baseline' },
    { name: 'Random Forest', accuracy: '87.13%', auc: '0.945', tag: 'Ensemble benchmark' },
  ];
}
