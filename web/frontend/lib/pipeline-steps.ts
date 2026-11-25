export const steps = [
  { title: 'Decode', text: 'Open the image, apply its camera orientation, and check the format and size.' },
  { title: 'Tile', text: 'Cut the receipt into overlapping patches at native resolution so small text stays visible.' },
  { title: 'Score', text: 'Score each patch with the frozen MobileNetV3 detector. Average the strongest tile scores and mark the tile with the highest score.' },
  { title: 'Calibrate', text: 'Map the document score to an estimated edit probability using the saved calibration fit.' },
  { title: 'Decide', text: 'Apply the frozen review rules. Low-confidence cases remain inconclusive and go to standard manual review.' },
];
