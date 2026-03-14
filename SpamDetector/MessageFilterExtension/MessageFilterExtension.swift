//
//  MessageFilterExtension.swift
//  MessageFilterExtension
//
//  Created by Mutti Atak on 14.03.2026.
//

import IdentityLookup

final class MessageFilterExtension: ILMessageFilterExtension {}

extension MessageFilterExtension: ILMessageFilterQueryHandling, ILMessageFilterCapabilitiesQueryHandling {
    func handle(_ capabilitiesQueryRequest: ILMessageFilterCapabilitiesQueryRequest, context: ILMessageFilterExtensionContext, completion: @escaping (ILMessageFilterCapabilitiesQueryResponse) -> Void) {
        let response = ILMessageFilterCapabilitiesQueryResponse()
        response.transactionalSubActions = [.transactionalOthers, .transactionalFinance, .transactionalOrders]
        response.promotionalSubActions = [.promotionalOthers, .promotionalOffers]
        completion(response)
    }

    func handle(_ queryRequest: ILMessageFilterQueryRequest, context: ILMessageFilterExtensionContext, completion: @escaping (ILMessageFilterQueryResponse) -> Void) {
        let (offlineAction, offlineSubAction) = self.offlineAction(for: queryRequest)

        switch offlineAction {
        case .allow, .junk, .promotion, .transaction:
            let response = ILMessageFilterQueryResponse()
            response.action = offlineAction
            response.subAction = offlineSubAction
            completion(response)

        case .none:
            // We handle everything offline — no network needed
            let response = ILMessageFilterQueryResponse()
            response.action = .allow
            completion(response)

        @unknown default:
            let response = ILMessageFilterQueryResponse()
            response.action = .allow
            completion(response)
        }
    }

    private func offlineAction(for queryRequest: ILMessageFilterQueryRequest) -> (ILMessageFilterAction, ILMessageFilterSubAction) {
        guard let messageBody = queryRequest.messageBody, !messageBody.isEmpty else {
            return (.allow, .none)
        }

        // Load classifier
        guard let classifier = SpamClassifier() else {
            // Fail-safe: if model can't load, allow everything
            NSLog("SpamDetector: Failed to load classifier")
            return (.allow, .none)
        }

        let prediction = classifier.predict(messageBody)

        if prediction.label == "spam" {
            // Categorize the spam type
            if prediction.signals.contains(where: { $0.contains("Phishing") }) {
                return (.junk, .none)
            } else if prediction.signals.contains(where: { $0.contains("spam keyword") || $0.contains("Discount") }) {
                return (.promotion, .promotionalOffers)
            } else {
                return (.junk, .none)
            }
        }

        return (.allow, .none)
    }

    private func networkAction(for networkResponse: ILNetworkResponse) -> (ILMessageFilterAction, ILMessageFilterSubAction) {
        // All classification is done offline — no network needed
        return (.none, .none)
    }
}
