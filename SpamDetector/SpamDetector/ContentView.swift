//
//  ContentView.swift
//  SpamDetector
//
//  Main UI — test the spam classifier + instructions to enable SMS filtering
//

import SwiftUI

struct ContentView: View {
    @State private var inputText = ""
    @State private var prediction: SpamClassifier.Prediction?

    private let classifier = SpamClassifier()

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 24) {
                    // Header
                    headerSection

                    // Enable filtering button
                    enableFilterSection

                    // Test classifier
                    testSection

                    // Result
                    if let prediction = prediction {
                        resultSection(prediction)
                    }

                    // Quick test buttons
                    quickTestSection
                }
                .padding()
            }
            .navigationTitle("Spam Detector")
        }
    }

    // MARK: - Header

    private var headerSection: some View {
        VStack(spacing: 8) {
            Image(systemName: "shield.checkered")
                .font(.system(size: 48))
                .foregroundStyle(.blue)

            Text("Turkish SMS Spam Filter")
                .font(.headline)

            Text("ML-powered spam detection for Turkish SMS messages")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding(.top)
    }

    // MARK: - Enable Filter Section

    private var enableFilterSection: some View {
        VStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 10) {
                Text("Enable SMS Filtering")
                    .font(.headline)

                setupStep(1, "Open Settings > Apps > Messages")
                setupStep(2, "Tap 'Unknown & Spam'")
                setupStep(3, "Enable 'Filter Unknown Senders'")
                setupStep(4, "Select 'SpamDetector' under SMS Filtering")
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private func setupStep(_ number: Int, _ text: String) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Text("\(number)")
                .font(.caption.bold())
                .foregroundStyle(.white)
                .frame(width: 24, height: 24)
                .background(Color.blue)
                .clipShape(Circle())

            Text(text)
                .font(.subheadline)
                .foregroundStyle(.primary)
        }
    }

    private func openSMSFilterSettings() {
        // iOS 18 blocks all prefs: and App-Prefs: URL schemes
        // Only openSettingsURLString works — opens Settings with our app selected
        if let url = URL(string: UIApplication.openSettingsURLString) {
            UIApplication.shared.open(url)
        }
    }

    // MARK: - Test Section

    private var testSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Test the Classifier")
                .font(.headline)

            TextEditor(text: $inputText)
                .frame(minHeight: 80)
                .padding(8)
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(Color(.systemGray4), lineWidth: 1)
                )
                .overlay(alignment: .topLeading) {
                    if inputText.isEmpty {
                        Text("Enter an SMS message to classify...")
                            .foregroundStyle(.tertiary)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 16)
                            .allowsHitTesting(false)
                    }
                }

            Button(action: classify) {
                HStack {
                    Image(systemName: "magnifyingglass")
                    Text("Classify")
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 12)
            }
            .buttonStyle(.borderedProminent)
            .disabled(inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        }
    }

    // MARK: - Result Section

    private func resultSection(_ prediction: SpamClassifier.Prediction) -> some View {
        VStack(spacing: 12) {
            HStack {
                Image(systemName: prediction.label == "spam" ? "xmark.shield.fill" : "checkmark.shield.fill")
                    .font(.title)
                    .foregroundStyle(prediction.label == "spam" ? .red : .green)

                VStack(alignment: .leading) {
                    Text(prediction.label == "spam" ? "SPAM" : "HAM (Legitimate)")
                        .font(.title2.bold())
                        .foregroundStyle(prediction.label == "spam" ? .red : .green)

                    Text("Confidence: \(prediction.confidence, specifier: "%.1f")%")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }

                Spacer()
            }

            if !prediction.signals.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Signals:")
                        .font(.caption.bold())
                        .foregroundStyle(.secondary)

                    ForEach(prediction.signals, id: \.self) { signal in
                        HStack(spacing: 4) {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .font(.caption2)
                                .foregroundStyle(.orange)
                            Text(signal)
                                .font(.caption)
                        }
                    }
                }
            }

            Text("Score: \(prediction.score, specifier: "%.4f")")
                .font(.caption)
                .foregroundStyle(.tertiary)
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(prediction.label == "spam"
                    ? Color.red.opacity(0.1)
                    : Color.green.opacity(0.1))
        )
    }

    // MARK: - Quick Test Buttons

    private var quickTestSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Quick Tests")
                .font(.headline)

            let examples: [(String, String)] = [
                ("Ham", "Aksam eve gelirken ekmek alir misin?"),
                ("Spam", "750 TL bonus kazanmak icin linke tikla! https://bit.ly/xyz"),
                ("Ham", "Faturanizin son odeme tarihi 15.03.2024'tur."),
                ("Spam", "BUYUK INDIRIM! %80 kampanya firsatini kacirmayin!"),
                ("Phishing", "Sn.BINANCE-TR Kullanicisi MASAK Tarafindan varliklariniz dondurulmustur"),
            ]

            ForEach(examples, id: \.1) { label, text in
                Button {
                    inputText = text
                    classify()
                } label: {
                    HStack {
                        Text(label)
                            .font(.caption.bold())
                            .foregroundStyle(.white)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(label == "Ham" ? Color.green : Color.red)
                            .clipShape(Capsule())

                        Text(text)
                            .font(.caption)
                            .lineLimit(1)
                            .foregroundStyle(.primary)

                        Spacer()
                    }
                }
            }
        }
    }

    // MARK: - Actions

    private func classify() {
        guard let classifier = classifier else { return }
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }

        withAnimation {
            var result = classifier.predict(text)
            result = SpamClassifier.Prediction(
                label: result.label,
                score: result.score,
                confidence: result.confidence * 100,
                signals: result.signals
            )
            prediction = result
        }
    }
}

#Preview {
    ContentView()
}
