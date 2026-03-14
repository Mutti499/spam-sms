//
//  SpamClassifier.swift
//  SpamDetector
//
//  Turkish SMS Spam Filter — Native Swift implementation
//  TF-IDF char n-gram vectorization + Linear SVM classifier
//

import Foundation

class SpamClassifier {

    // MARK: - Model Data

    struct FeatureInfo: Codable {
        let i: Int      // IDF index
        let w: Double   // SVM weight
    }

    struct ModelData: Codable {
        let version: Int
        let type: String
        let analyzer: String
        let ngram_range: [Int]
        let sublinear_tf: Bool
        let intercept: Double
        let idf: [Double]
        let features: [String: FeatureInfo]
    }

    struct SpamConfig: Codable {
        let spam_keywords: [String]
        let opt_out_patterns: [String]
        let phishing_domains: [String]
        let phishing_keywords: [String]
        let url_shorteners: [String]
    }

    // MARK: - Properties

    private let modelData: ModelData
    private let config: SpamConfig

    // MARK: - Initialization

    init?() {
        guard let modelURL = Bundle.main.url(forResource: "spam_model", withExtension: "json"),
              let configURL = Bundle.main.url(forResource: "spam_config", withExtension: "json") else {
            print("SpamClassifier: Model files not found in bundle")
            return nil
        }

        do {
            let modelJSON = try Data(contentsOf: modelURL)
            self.modelData = try JSONDecoder().decode(ModelData.self, from: modelJSON)

            let configJSON = try Data(contentsOf: configURL)
            self.config = try JSONDecoder().decode(SpamConfig.self, from: configJSON)

            print("SpamClassifier loaded: \(modelData.features.count) features")
        } catch {
            print("SpamClassifier init error: \(error)")
            return nil
        }
    }

    // Also support loading from a specific path (for app extensions)
    init?(modelPath: String, configPath: String) {
        let modelURL = URL(fileURLWithPath: modelPath)
        let configURL = URL(fileURLWithPath: configPath)

        do {
            let modelJSON = try Data(contentsOf: modelURL)
            self.modelData = try JSONDecoder().decode(ModelData.self, from: modelJSON)

            let configJSON = try Data(contentsOf: configURL)
            self.config = try JSONDecoder().decode(SpamConfig.self, from: configJSON)
        } catch {
            print("SpamClassifier init error: \(error)")
            return nil
        }
    }

    // MARK: - Prediction

    struct Prediction {
        let label: String        // "spam" or "ham"
        let score: Double        // SVM decision score
        let confidence: Double   // 0.0 - 1.0
        let signals: [String]    // Human-readable reasons
    }

    func predict(_ text: String) -> Prediction {
        var signals: [String] = []

        // Step 1: Rule-based pre-filter (instant detection)
        let ruleResult = applyRules(text)
        if let forced = ruleResult.forced {
            return Prediction(
                label: forced,
                score: forced == "spam" ? 1.0 : -1.0,
                confidence: 0.99,
                signals: ruleResult.signals
            )
        }
        signals.append(contentsOf: ruleResult.signals)

        // Step 2: ML classification
        let mlScore = computeMLScore(text)
        let label = mlScore > 0 ? "spam" : "ham"

        // Sigmoid-based confidence
        let confidence = 1.0 / (1.0 + exp(-abs(mlScore) * 3.0))

        return Prediction(
            label: label,
            score: mlScore,
            confidence: confidence,
            signals: signals
        )
    }

    // MARK: - ML Scoring (TF-IDF + Linear SVM)

    private func computeMLScore(_ text: String) -> Double {
        let ngramMin = modelData.ngram_range[0]
        let ngramMax = modelData.ngram_range[1]

        // Pad text for char_wb analyzer
        let padded = " \(text) "
        let chars = Array(padded)

        // Extract character n-grams and count occurrences
        var ngramCounts: [String: Int] = [:]
        for n in ngramMin...ngramMax {
            for i in 0...(chars.count - n) {
                let ngram = String(chars[i..<(i + n)])
                if modelData.features[ngram] != nil {
                    ngramCounts[ngram, default: 0] += 1
                }
            }
        }

        // Compute TF-IDF values
        var tfidfVals: [String: Double] = [:]
        for (ngram, count) in ngramCounts {
            guard let feat = modelData.features[ngram] else { continue }
            // sublinear_tf: tf = 1 + log(count)
            let tf = modelData.sublinear_tf ? 1.0 + log(Double(count)) : Double(count)
            tfidfVals[ngram] = tf * modelData.idf[feat.i]
        }

        // L2 normalize
        let norm = sqrt(tfidfVals.values.reduce(0.0) { $0 + $1 * $1 })
        if norm > 0 {
            for ngram in tfidfVals.keys {
                tfidfVals[ngram]! /= norm
            }
        }

        // Dot product with SVM weights
        var score = modelData.intercept
        for (ngram, tfidfVal) in tfidfVals {
            score += tfidfVal * modelData.features[ngram]!.w
        }

        return score
    }

    // MARK: - Rule-based Pre-filter

    private struct RuleResult {
        var forced: String?     // nil = let ML decide
        var signals: [String]
    }

    private func applyRules(_ text: String) -> RuleResult {
        let lower = text.lowercased()
        var signals: [String] = []

        // Check phishing domains (always spam)
        for domain in config.phishing_domains {
            if lower.contains(domain.lowercased()) {
                return RuleResult(forced: "spam", signals: ["Phishing domain: \(domain)"])
            }
        }

        // Check phishing keywords (always spam)
        for keyword in config.phishing_keywords {
            if lower.contains(keyword.lowercased()) {
                return RuleResult(forced: "spam", signals: ["Phishing detected: \(keyword)"])
            }
        }

        // Check opt-out patterns (strong spam signal)
        for pattern in config.opt_out_patterns {
            if lower.contains(pattern.lowercased()) {
                signals.append("Has opt-out text")
                break
            }
        }

        // Count spam keywords
        var keywordCount = 0
        for keyword in config.spam_keywords {
            if lower.contains(keyword) {
                keywordCount += 1
            }
        }
        if keywordCount > 0 {
            signals.append("\(keywordCount) spam keyword(s)")
        }

        // URL shortener check
        for shortener in config.url_shorteners {
            if lower.contains(shortener.lowercased()) {
                signals.append("URL shortener detected")
                break
            }
        }

        // Discount pattern
        if lower.range(of: #"%\d+"#, options: .regularExpression) != nil {
            signals.append("Discount pattern (%)")
        }

        return RuleResult(forced: nil, signals: signals)
    }
}
