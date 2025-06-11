"""Module pour les opérations de scan réseau avec rotation d'IP."""

import asyncio
import re
import socket
import ssl
import logging
import aiohttp
import certifi
import random
import time
import shutil
from typing import List, Dict, Optional, Set
from scapy.all import IP, TCP, UDP, ICMP, sr1, RandShort
from datetime import datetime
import dns.resolver
import torch
from network_tool.config import APP_CONFIG
from network_tool.state import ScannerState
from network_tool.models import LSTMModel  # Import du modèle LSTM

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Ajustement des paramètres de configuration pour répondre aux recommandations
APP_CONFIG.MAX_PARALLEL = 50  # Réduit pour limiter la surcharge réseau
APP_CONFIG.TIMEOUT = 2  # Augmenté à 2 secondes pour permettre des réponses lentes
APP_CONFIG.RETRY_BACKOFF_FACTOR = 2  # Facteur de backoff pour les tentatives
APP_CONFIG.MAX_RETRIES = 3  # Nombre maximum de tentatives
APP_CONFIG.USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Mac OSX 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "ShopifyBot/1.0 (+https://shopify.com)"
]

class NetworkScanner:
    """
    Classe pour effectuer des scans réseau avancés avec rotation d'IP et diverses techniques de détection.

    Cette classe encapsule des fonctionnalités pour scanner des ports, identifier des services,
    détecter des protections (comme les WAFs ou Cloudflare), énumérer des sous-domaines,
    et vérifier des vulnérabilités. Elle utilise `asyncio` pour les opérations réseau
    asynchrones, `scapy` pour la manipulation de paquets, et peut intégrer un modèle LSTM
    pour la prédiction de services. La rotation d'IP source et de proxies est supportée
    pour améliorer la furtivité et contourner certaines protections.

    Attributes:
        state (ScannerState): L'état partagé du scanner, contenant par exemple les droits admin.
        semaphore (asyncio.Semaphore): Pour limiter le nombre d'opérations parallèles.
        config (APP_CONFIG): Configuration de l'application (timeouts, proxies, etc.).
        lstm_model (Optional[LSTMModel]): Modèle LSTM pour la prédiction de services.
        service_signatures (Dict): Signatures heuristiques pour l'identification de services.
        response_times (List[float]): Liste pour stocker les temps de réponse des paquets.
        proxies (List[Dict[str, str]]): Liste des proxies à utiliser pour les requêtes HTTP/HTTPS.
        current_proxy_index (int): Index du proxy actuel dans la liste.
        source_ips (List[str]): Liste des adresses IP source à utiliser pour Scapy.
        packet_loss_count (int): Compteur pour les pertes de paquets.
    """
    
    def __init__(self, state: ScannerState) -> None:
        """
        Initialise le NetworkScanner.

        Args:
            state (ScannerState): L'objet d'état du scanner, qui doit inclure
                                  les attributs 'is_admin' (bool) et 'executor'
                                  (concurrent.futures.Executor).

        Raises:
            ValueError: Si l'objet `state` ne contient pas les attributs requis.
        """
        if not hasattr(state, 'is_admin'):
            raise ValueError("L'objet state doit contenir l'attribut 'is_admin'")
        if not hasattr(state, 'executor'):
            raise ValueError("L'objet state doit contenir l'attribut 'executor'")
        self.state: ScannerState = state
        self.config = APP_CONFIG  # self.config est maintenant une référence à APP_CONFIG
        self.semaphore: asyncio.Semaphore = asyncio.Semaphore(self.config.MAX_PARALLEL) # Utiliser self.config ici
        self.lstm_model: Optional[LSTMModel] = self._load_lstm_model()
        self.service_signatures: Dict[str, Dict] = {
            "ssh": {"window": [0, 65535], "ttl": [64, 128], "flags": ["SA", "R"], "confidence": 0.7},
            "http": {"window": [8192, 32768], "ttl": [64, 128], "flags": ["SA", "R"], "confidence": 0.6},
            "firewall": {"icmp_type": 3, "confidence": 0.8}
        }
        self.response_times: List[float] = []
        self.proxies: List[Dict[str, str]] = self._load_proxies()
        self.current_proxy_index: int = 0
        self.source_ips: List[str] = self.config.SOURCE_IPS if hasattr(self.config, 'SOURCE_IPS') else [] # Liste d'IP sources pour spoofing
        self.packet_loss_count: int = 0 # Compteur global de paquets perdus

    def _load_lstm_model(self) -> Optional[LSTMModel]:
        """
        Charge le modèle LSTM pour la prédiction de services.

        Tente de charger un modèle pré-entraîné depuis 'data.pkl'.
        Si le fichier n'est pas trouvé ou en cas d'erreur, un nouveau modèle
        est initialisé avec des poids aléatoires. Le modèle est déplacé sur GPU
        si CUDA est disponible.

        Returns:
            Optional[LSTMModel]: Le modèle LSTM chargé ou initialisé, ou None en cas d'erreur majeure.
        """
        try:
            model: LSTMModel = LSTMModel(input_size=6, hidden_size=128, num_layers=2, output_size=10) # Initialisation de l'architecture du modèle
            model_path: str = "data.pkl" # Chemin vers le fichier de poids pré-entraînés

            # Charger le modèle sur GPU si disponible, sinon CPU
            if torch.cuda.is_available():
                model = model.cuda()
                state_dict = torch.load(model_path, map_location=torch.device('cuda')) # Charger les poids pour CUDA
            else:
                state_dict = torch.load(model_path, map_location=torch.device('cpu')) # Charger les poids pour CPU

            model.load_state_dict(state_dict) # Appliquer les poids chargés au modèle
            model.eval() # Mettre le modèle en mode évaluation (désactive dropout, etc.)
            logger.info(f"Modèle LSTM chargé avec succès depuis {model_path}")
            return model
        except FileNotFoundError:
            # Si le fichier de poids n'est pas trouvé, initialiser un nouveau modèle (non entraîné)
            logger.error(f"Échec du chargement des poids du modèle LSTM : Fichier {model_path} introuvable. Initialisation avec des poids aléatoires.")
            model: LSTMModel = LSTMModel(input_size=6, hidden_size=128, num_layers=2, output_size=10)
            if torch.cuda.is_available():
                model = model.cuda()
            model.eval()
            return model
        except Exception as e:
            # Gérer toute autre erreur pendant le chargement
            logger.error(f"Erreur inattendue lors du chargement du modèle LSTM : {e}", exc_info=True)
            return None

    def _load_proxies(self) -> List[Dict[str, str]]:
        """
        Charge la liste des proxies depuis la configuration.

        Utilise `APP_CONFIG.PROXIES` si disponible, sinon une liste par défaut.
        Affiche un avertissement si aucun proxy n'est configuré.

        Returns:
            List[Dict[str, str]]: Une liste de dictionnaires de proxies, où chaque
                                   dictionnaire contient les clés "http" et "https".
                                   Retourne une liste vide en cas d'erreur.
        """
        try:
            # Tenter de récupérer la liste des proxies depuis la configuration
            proxies: List[Dict[str, str]] = getattr(self.config, 'PROXIES', [
                # Liste de proxies par défaut si APP_CONFIG.PROXIES n'existe pas
                {"http": "http://proxy1.example.com:8080", "https": "https://proxy1.example.com:8080"},
                {"http": "http://proxy2.example.com:8080", "https": "https://proxy2.example.com:8080"},
            ])
            if not proxies:
                # Avertir si aucune configuration de proxy n'est trouvée
                logger.warning("Aucun proxy configuré, rotation d'IP désactivée pour les requêtes HTTP/HTTPS")
            return proxies
        except Exception as e:
            logger.error(f"Erreur lors du chargement des proxies : {e}", exc_info=True)
            return [] # Retourner une liste vide en cas d'erreur

    def _get_next_proxy(self) -> Optional[Dict[str, str]]:
        """
        Fournit le prochain proxy de la liste de manière circulaire.

        Returns:
            Optional[Dict[str, str]]: Le prochain proxy à utiliser, ou None si la liste est vide.
        """
        if not self.proxies: # Si la liste des proxies est vide
            return None
        # Sélectionne un proxy en utilisant l'index actuel et l'opérateur modulo pour une rotation circulaire
        proxy: Dict[str, str] = self.proxies[self.current_proxy_index % len(self.proxies)]
        self.current_proxy_index += 1 # Incrémenter l'index pour la prochaine demande
        logger.debug(f"Utilisation du proxy : {proxy}")
        return proxy

    def _get_random_source_ip(self) -> Optional[str]:
        """
        Sélectionne une adresse IP source aléatoire parmi celles configurées.

        Utilisé pour la construction de paquets Scapy afin de varier l'origine des sondes.

        Returns:
            Optional[str]: Une adresse IP source aléatoire, ou None si aucune n'est configurée.
        """
        if not self.source_ips: # Si aucune IP source n'est configurée
            return None
        # Choisir une IP aléatoire dans la liste des IPs sources disponibles
        source_ip: str = random.choice(self.source_ips)
        logger.debug(f"Utilisation de l'IP source : {source_ip}")
        return source_ip

    def _extract_features(self, pkt: IP, response_time: float) -> torch.Tensor:
        """
        Extrait les caractéristiques d'un paquet réseau pour l'alimentation du modèle LSTM.

        Args:
            pkt (IP): Le paquet Scapy (couche IP) reçu.
            response_time (float): Le temps de réponse pour obtenir ce paquet.

        Returns:
            torch.Tensor: Un tenseur contenant les caractéristiques extraites.
                          Retourne un tenseur par défaut en cas d'erreur.
        """
        try:
            # Extraction des caractéristiques: TTL, taille de fenêtre TCP, longueur du paquet IP, temps de réponse, présence ICMP, nombre d'options TCP.
            # Des valeurs par défaut sont utilisées si une caractéristique ne peut être extraite.
            features: List[float] = [
                float(pkt[IP].ttl) if pkt and pkt.haslayer(IP) else 64.0,  # Time-To-Live du paquet IP
                float(pkt[TCP].window) if pkt and pkt.haslayer(TCP) else 0.0,  # Taille de la fenêtre TCP
                float(pkt[IP].len) if pkt and pkt.haslayer(IP) else 0.0,  # Longueur totale du paquet IP
                response_time,  # Temps de réponse pour recevoir le paquet
                1.0 if pkt and pkt.haslayer(ICMP) else 0.0,  # Indicateur binaire pour la présence d'ICMP
                float(len(pkt[TCP].options)) if pkt and pkt.haslayer(TCP) and hasattr(pkt[TCP], 'options') else 0.0  # Nombre d'options TCP
            ]
            return torch.tensor([features], dtype=torch.float32) # Conversion en tenseur PyTorch
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction des caractéristiques : {e}", exc_info=True)
            # Retourner un tenseur de caractéristiques par défaut en cas d'erreur
            return torch.tensor([[64.0, 0.0, 0.0, response_time, 0.0, 0.0]], dtype=torch.float32)

    def _predict_service_lstm(self, features: torch.Tensor) -> tuple[str, float]:
        """
        Prédit le service réseau en utilisant le modèle LSTM.

        Args:
            features (torch.Tensor): Le tenseur de caractéristiques extrait du paquet.

        Returns:
            tuple[str, float]: Un tuple contenant le nom du service prédit et le score de confiance.
                               Retourne ("unknown", 0.5) si le modèle LSTM n'est pas chargé ou en cas d'erreur.
        """
        if not self.lstm_model: # Vérifier si le modèle LSTM est chargé
            return "unknown", 0.5
        try:
            with torch.no_grad(): # Désactiver le calcul du gradient pour l'inférence
                output: torch.Tensor = self.lstm_model(features) # Obtenir la sortie brute du modèle
                # Appliquer softmax pour obtenir des probabilités et prendre l'argmax pour la classe prédite
                service_idx: int = torch.argmax(output, dim=1).item() # Index du service avec la plus haute probabilité
                confidence: float = torch.softmax(output, dim=1)[0][service_idx].item() # Score de confiance pour le service prédit

                # Liste des services correspondant aux indices de sortie du modèle
                services: List[str] = ["http", "ssh", "ftp", "smtp", "telnet", "dns", "mysql", "rdp", "sip", "unknown"]
                # S'assurer que l'index est dans les limites de la liste des services
                return services[min(service_idx, len(services)-1)], confidence
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction LSTM : {e}", exc_info=True)
            return "unknown", 0.5 # Retour par défaut en cas d'erreur

    def _predict_service_heuristic(self, pkt: IP, response_time: float) -> tuple[str, float]:
        """
        Prédit le service réseau en utilisant des règles heuristiques basées sur les paquets.

        Args:
            pkt (IP): Le paquet Scapy (couche IP) reçu.
            response_time (float): Le temps de réponse (non utilisé directement dans cette heuristique précise,
                                   mais conservé pour cohérence d'interface si besoin futur).

        Returns:
            tuple[str, float]: Un tuple contenant le nom du service prédit et le score de confiance.
                               Retourne ("unknown", 0.5) si aucune règle ne correspond ou en cas d'erreur.
        """
        try:
            # Si un paquet ICMP de type 3 (Destination Unreachable) est reçu, le port est probablement filtré
            if pkt and pkt.haslayer(ICMP) and pkt[ICMP].type == 3:
                return "filtered", self.service_signatures["firewall"]["confidence"]

            # Analyse des paquets TCP pour des signatures de service connues
            if pkt and pkt.haslayer(TCP):
                window: int = pkt[TCP].window # Taille de la fenêtre TCP
                ttl: int = pkt[IP].ttl if pkt.haslayer(IP) else 64 # TTL du paquet IP
                flags: str = str(pkt[TCP].flags) # Flags TCP (ex: "SA" pour SYN-ACK)

                # Comparer avec les signatures de service stockées
                for service, sig in self.service_signatures.items():
                    if service == "firewall": # Ignorer la signature de pare-feu ici, gérée par ICMP
                        continue
                    # Vérifier si les caractéristiques du paquet correspondent à une signature
                    if (sig["window"][0] <= window <= sig["window"][1] and
                        sig["ttl"][0] <= ttl <= sig["ttl"][1] and
                        flags in sig["flags"]):
                        return service, sig["confidence"] # Retourner le service et la confiance si une correspondance est trouvée

            return "unknown", 0.5 # Aucun service identifié par heuristique
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction heuristique : {e}", exc_info=True)
            return "unknown", 0.5 # Retour par défaut en cas d'erreur

    async def resolve_domain(self, domain: str) -> Set[str]:
        """
        Résout un nom de domaine en un ensemble d'adresses IP (enregistrements A).

        Args:
            domain (str): Le nom de domaine à résoudre.

        Returns:
            Set[str]: Un ensemble d'adresses IP associées au domaine.
                      Retourne un ensemble vide en cas d'erreur ou si aucune IP n'est trouvée.
        """
        ip_set: Set[str] = set()
        try:
            # Exécute la résolution DNS dans un thread séparé pour ne pas bloquer la boucle d'événements asyncio
            # `dns.resolver.resolve` est une opération bloquante.
            answers: Any = await asyncio.get_event_loop().run_in_executor(None, lambda: dns.resolver.resolve(domain, 'A'))
            for rdata in answers: # Parcourir les réponses DNS
                ip_set.add(rdata.address) # Ajouter chaque adresse IP résolue à l'ensemble

            logger.debug(f"Résolution DNS réussie pour {domain} : {ip_set}")
            if not ip_set: # Si aucune IP n'a été résolue
                logger.warning(f"Aucune adresse IP résolue pour {domain}. Vérifiez le domaine ou utilisez une IP directe.")
            else:
                logger.info(f"Adresses IP résolues pour {domain} : {', '.join(ip_set)}")
        except dns.resolver.NXDOMAIN: # Le domaine n'existe pas
            logger.warning(f"Domaine {domain} non trouvé.")
        except dns.resolver.NoAnswer: # Le domaine existe mais n'a pas d'enregistrement A
            logger.warning(f"Aucune réponse DNS pour {domain}.")
        except Exception as e: # Autres erreurs de résolution DNS
            logger.error(f"Erreur lors de la résolution DNS de {domain} : {e}", exc_info=True)
        return ip_set

    async def tcp_connect(self, ip: str, port: int) -> Dict[str, Any]:
        """
        Tente une connexion TCP standard à une adresse IP et un port spécifiés.

        Cette méthode est utilisée comme un scan de base pour vérifier si un port est ouvert.
        Elle effectue plusieurs tentatives en cas d'échec initial.

        Args:
            ip (str): L'adresse IP cible.
            port (int): Le port cible.

        Returns:
            Dict[str, Any]: Un dictionnaire contenant les résultats du scan:
                - "status" (str): "open", "closed", ou "filtered".
                - "service" (str): Le service détecté (souvent "unknown" à ce stade).
                - "confidence" (float): Score de confiance de la détection.
                - "protection" (str): Protection détectée ("none" par défaut ici).
                - "os_guess" (str): Estimation de l'OS (peut être affinée par d'autres méthodes).
        """
        async with self.semaphore:
            results: Dict[str, Any] = {"status": "closed", "service": "unknown", "confidence": 0.5, "protection": "none", "os_guess": "unknown"}
            # Boucle pour plusieurs tentatives de connexion
            for attempt in range(self.config.MAX_RETRIES):
                try:
                    # Tenter d'établir une connexion avec un timeout
                    async with asyncio.timeout(self.config.TIMEOUT):
                        reader, writer = await asyncio.open_connection(ip, port) # type: ignore
                        # Si la connexion réussit, le port est ouvert
                        results.update({"status": "open", "confidence": 0.9, "protection": "none"})
                        writer.close() # Fermer la connexion
                        await writer.wait_closed() # Attendre que la fermeture soit complète
                        logger.info(f"Scan TCP connect : {ip}:{port} ouvert (tentative {attempt+1})")
                        return results # Retourner immédiatement si ouvert
                except (asyncio.TimeoutError, ConnectionRefusedError, OSError) as e:
                    # Gérer les erreurs courantes indiquant un port fermé ou filtré
                    logger.debug(f"Scan TCP connect : {ip}:{port} fermé ou filtré (tentative {attempt+1}) : {e}")
                    self.packet_loss_count += 1 # Incrémenter le compteur de pertes de paquets/erreurs
                    if attempt < self.config.MAX_RETRIES - 1:
                        # Attendre un court instant avant la prochaine tentative
                        await asyncio.sleep(random.uniform(0.1, 0.5))

            # Si toutes les tentatives de connexion TCP standard échouent,
            # utiliser une méthode de sonde plus avancée (SYN scan like) pour affiner le diagnostic.
            closed_results: Dict[str, Any] = await self.probe_closed_port_advanced(ip, port)
            results.update(closed_results) # Mettre à jour les résultats avec ceux de la sonde avancée

            # Correction: si la sonde avancée retourne un statut "error", le considérer comme "closed" par défaut.
            if results["status"] == "error":
                results["status"] = "closed"
                logger.warning(f"Statut 'error' corrigé en 'closed' pour {ip}:{port}")
            return results

    async def probe_closed_port_advanced(self, ip: str, port: int) -> Dict[str, Any]:
        """
        Effectue une sonde avancée sur un port présumé fermé ou filtré.

        Utilise des paquets SYN Scapy pour obtenir plus d'informations (par exemple,
        distinguer un port fermé d'un port filtré, ou détecter un service inattendu).
        Peut également prédire le service en utilisant des heuristiques ou un modèle LSTM.

        Args:
            ip (str): L'adresse IP cible.
            port (int): Le port cible.

        Returns:
            Dict[str, Any]: Un dictionnaire de résultats similaire à `tcp_connect`,
                            mais potentiellement avec des informations plus précises
                            sur le statut du port, le service, ou l'OS.
        """
        async with self.semaphore: # Contrôle de concurrence
            results: Dict[str, Any] = {"status": "closed", "service": "unknown", "confidence": 0.5, "protection": "none", "os_guess": "unknown"}

        # Multiples tentatives pour la sonde avancée
        for attempt in range(self.config.MAX_RETRIES):
            try:
                source_ip: Optional[str] = self._get_random_source_ip() # Rotation d'IP source si configurée
                # Construction du paquet SYN avec Scapy. `sport=RandShort()` utilise un port source aléatoire.
                pkt: Any = IP(dst=ip, src=source_ip) / TCP(dport=port, flags="S", sport=RandShort()) if source_ip else IP(dst=ip) / TCP(dport=port, flags="S", sport=RandShort())

                start_time: float = time.time()
                # Envoyer le paquet et attendre une réponse (`sr1` envoie et reçoit un seul paquet)
                # Exécuté dans un thread pour ne pas bloquer asyncio.
                response: Optional[IP] = await asyncio.get_event_loop().run_in_executor( # type: ignore
                    None, lambda: sr1(pkt, timeout=self.config.TIMEOUT, verbose=0) # `verbose=0` pour moins de logs Scapy
                )
                response_time: float = time.time() - start_time # Calcul du temps de réponse
                self.response_times.append(response_time)
                logger.debug(f"Sonde SYN sur {ip}:{port} : réponse en {response_time:.3f}s")

                if response:
                    # Analyser la réponse pour déterminer le statut du port
                    if response.haslayer(TCP) and response[TCP].flags == "SA":  # SYN-ACK (flag 0x12) indique un port ouvert
                        results.update({"status": "open", "confidence": 0.95})
                        # Tenter de prédire le service basé sur la réponse
                        service, confidence = self._predict_service_heuristic(response, response_time)
                        if self.lstm_model: # Si un modèle LSTM est disponible, l'utiliser pour affiner la prédiction
                            features: torch.Tensor = self._extract_features(response, response_time)
                            lstm_service, lstm_confidence = self._predict_service_lstm(features)
                            if lstm_confidence > confidence: # Choisir la prédiction avec la plus haute confiance
                                service, confidence = lstm_service, lstm_confidence
                        results.update({"service": service, "confidence": confidence})
                        os_guess: str = await self.fingerprint_tcp_ip(response) # Fingerprinting de l'OS
                        results["os_guess"] = os_guess
                        logger.info(f"Port {ip}:{port} détecté comme ouvert avec service {service}")
                        return results # Port trouvé ouvert, retourner
                    elif response.haslayer(TCP) and response[TCP].flags == "R":  # RST (flag 0x04 ou 0x14 pour RST-ACK) indique un port fermé
                        results.update({"status": "closed", "confidence": 0.9})
                        logger.debug(f"Port {ip}:{port} détecté comme fermé (RST)")
                    elif response.haslayer(ICMP) and response[ICMP].type == 3:  # ICMP Destination Unreachable (type 3) indique un port filtré
                        results.update({"status": "filtered", "service": "filtered", "confidence": 0.7, "protection": "firewall"})
                        logger.debug(f"Port {ip}:{port} détecté comme filtré (ICMP type 3)")
                    else: # Réponse inattendue, souvent signe de filtrage ou d'un comportement réseau non standard
                        results.update({"status": "filtered", "confidence": 0.7})
                        logger.debug(f"Port {ip}:{port} détecté comme filtré (réponse inattendue: {response.summary()})")
                else: # Aucune réponse au paquet SYN
                    results.update({"status": "filtered", "confidence": 0.7}) # Indique souvent un port filtré (ou "ouvert|filtré")
                    self.packet_loss_count += 1
                    logger.debug(f"Port {ip}:{port} : aucune réponse, possible perte de paquet")

                # Tenter une sonde UDP si le port est un port UDP commun, même si une réponse TCP a été reçue.
                # Cela peut être utile si un service UDP écoute sur le même numéro de port.
                udp_results: Dict[str, Any] = await self._send_udp_probe(ip, port)
                if udp_results["service"] != "unknown":
                    # Si un service UDP est détecté, cela peut être plus pertinent que l'état TCP "closed" ou "filtered".
                    results.update(udp_results)
                    logger.info(f"Port {ip}:{port} : service UDP détecté - {udp_results['service']}")
                    return results

                return results # Retourner les résultats de la tentative actuelle si aucune sonde UDP n'a abouti
            except Exception as e: # Gérer les erreurs pendant la sonde
                logger.error(f"Erreur lors de la sonde avancée sur {ip}:{port} (tentative {attempt+1}) : {e}", exc_info=True)
                self.packet_loss_count += 1
                if attempt < self.config.MAX_RETRIES - 1: # Attendre avant la prochaine tentative
                    await asyncio.sleep(random.uniform(0.1, 0.5))

        # Si toutes les tentatives échouent
        logger.warning(f"Scan avancé échoué pour {ip}:{port} après {self.config.MAX_RETRIES} tentatives")
        return results

    async def _send_udp_probe(self, ip: str, port: int) -> Dict[str, Any]:
        """
        Envoie des sondes UDP spécifiques à un port pour identifier des services communs (DNS, SNMP, SIP).

        Args:
            ip (str): L'adresse IP cible.
            port (int): Le port UDP cible.

        Returns:
            Dict[str, Any]: Un dictionnaire contenant:
                - "service" (str): Le service UDP détecté ("dns", "snmp", "sip", ou "unknown").
                - "confidence" (float): Le score de confiance de la détection.
        """
        async with self.semaphore: # Contrôle de concurrence
            results: Dict[str, Any] = {"service": "unknown", "confidence": 0.5}
        try:
            # Dictionnaire de payloads spécifiques pour différents protocoles UDP
            payloads: Dict[str, bytes] = {
                "dns": b"\x00\x01\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00\x07example\x03com\x00\x00\x01\x00\x01", # Requête DNS standard pour example.com
                "snmp": b"\x30\x26\x02\x01\x00\x04\x06\x70\x75\x62\x6c\x69\x63\xa0\x19\x02\x04\x00\x00\x00\x01\x02\x01\x00\x02\x01\x00\x30\x0b\x30\x09\x06\x05\x2b\x06\x01\x02\x01\x01\x00", # Requête SNMP GetRequest
                "sip": b"OPTIONS sip:user@shopify.com SIP/2.0\r\nVia: SIP/2.0/UDP 127.0.0.1\r\n\r\n" # Requête SIP OPTIONS
            }
            # Itérer sur chaque protocole et son payload associé
            for proto, payload_bytes in payloads.items():
                source_ip: Optional[str] = self._get_random_source_ip() # Rotation IP source
                # Construire le paquet UDP avec Scapy
                pkt: Any = IP(dst=ip, src=source_ip) / UDP(dport=port, sport=RandShort()) / payload_bytes if source_ip else IP(dst=ip) / UDP(dport=port, sport=RandShort()) / payload_bytes

                start_time: float = time.time()
                # Envoyer le paquet UDP et attendre une réponse
                response: Optional[IP] = await asyncio.get_event_loop().run_in_executor( # type: ignore
                    None, lambda: sr1(pkt, timeout=self.config.TIMEOUT, verbose=0) # `verbose=0` pour Scapy
                )
                response_time: float = time.time() - start_time # Calculer le temps de réponse
                self.response_times.append(response_time)
                await asyncio.sleep(random.uniform(0.1, 0.5))  # Délai aléatoire
                if response and response.haslayer(UDP):
                    results.update({"service": proto, "confidence": 0.9})
                    logger.debug(f"Sonde UDP {proto} sur {ip}:{port} : réponse UDP reçue")
                    break
                elif response and response.haslayer(ICMP) and response[ICMP].type == 3:
                    logger.debug(f"Sonde UDP {proto} sur {ip}:{port} : ICMP Port Unreachable")
                else:
                    self.packet_loss_count += 1
                    logger.debug(f"Sonde UDP {proto} sur {ip}:{port} : aucune réponse ou réponse non-UDP/ICMP")
        except Exception as e:
            logger.error(f"Erreur lors de la sonde UDP sur {ip}:{port} : {e}", exc_info=True)
            self.packet_loss_count += 1
        return results

    async def fingerprint_tcp_ip(self, pkt: IP) -> str:
        """
        Tente de deviner le système d'exploitation (OS) basé sur le TTL d'un paquet IP.

        Args:
            pkt (IP): Le paquet Scapy (couche IP) reçu en réponse.

        Returns:
            str: Une chaîne représentant l'OS deviné ("linux", "windows", "cisco", "cloudflare", ou "unknown").
        """
        try:
            if not pkt or not pkt.haslayer(IP): # S'assurer que le paquet est valide et contient une couche IP
                return "unknown"
            ttl: int = pkt[IP].ttl # Extraire le Time-To-Live (TTL)

            # Dictionnaire de signatures TTL pour différents OS (valeurs approximatives)
            os_signatures: Dict[str, Dict[str, List[int]]] = {
                "linux": {"ttl": [60, 64]},       # Linux utilise souvent un TTL initial de 64
                "windows": {"ttl": [120, 128]},   # Windows utilise souvent un TTL initial de 128
                "cisco": {"ttl": [250, 255]},     # Équipements Cisco peuvent avoir un TTL élevé
                "cloudflare": {"ttl": [110, 120]} # TTL observé pour des serveurs derrière Cloudflare
            }
            # Comparer le TTL du paquet avec les signatures connues
            for os_name, sig in os_signatures.items():
                if sig["ttl"][0] <= ttl <= sig["ttl"][1]: # Si le TTL est dans la plage d'une signature
                    logger.debug(f"OS détecté : {os_name} (TTL={ttl})")
                    return os_name # Retourner le nom de l'OS

            return "unknown" # Si le TTL ne correspond à aucune signature connue
        except Exception as e:
            logger.error(f"Erreur lors de l'empreinte TCP/IP : {e}", exc_info=True)
            return "unknown" # Retourner "unknown" en cas d'erreur

    async def syn_scan(self, ip: str, port: int) -> Dict[str, Any]:
        """
        Effectue un scan SYN (half-open scan) sur une IP et un port.

        Nécessite des privilèges administratifs pour forger des paquets SYN bruts.
        Si non disponible, retombe sur un scan TCP connect.
        Analyse la réponse pour déterminer si le port est ouvert, fermé ou filtré.

        Args:
            ip (str): L'adresse IP cible.
            port (int): Le port cible.

        Returns:
            Dict[str, Any]: Un dictionnaire de résultats détaillés, incluant:
                - "status": "open", "closed", "filtered".
                - "service": Service prédit (LSTM ou heuristique).
                - "confidence": Score de confiance.
                - "bypass_method": "syn" si le scan SYN a réussi.
                - "protection": Protection détectée (ex: "firewall").
                - "os_guess": Estimation de l'OS.
        """
        async with self.semaphore:
            results: Dict[str, Any] = {"status": "closed", "service": "unknown", "confidence": 0.5, "bypass_method": "none", "protection": "none", "os_guess": "unknown"}
        if not self.state.is_admin:
            logger.warning(f"Scan SYN ignoré pour {ip}:{port} : privilèges administratifs requis. Passage à TCP connect.")
            return await self.tcp_connect(ip, port)
        try:
            source_ip: Optional[str] = self._get_random_source_ip()
            packet: Any = IP(dst=ip, src=source_ip) / TCP(dport=port, flags="S", sport=RandShort()) if source_ip else IP(dst=ip) / TCP(dport=port, flags="S", sport=RandShort())
            start_time: float = time.time()
            response: Optional[IP] = await asyncio.get_event_loop().run_in_executor( # type: ignore
                None, lambda: sr1(packet, timeout=self.config.TIMEOUT, verbose=0)
            )
            response_time: float = time.time() - start_time
            self.response_times.append(response_time)
            await asyncio.sleep(random.uniform(0.1, 0.5))  # Délai aléatoire
            if response:
                logger.debug(f"Réponse SYN reçue pour {ip}:{port} : {response.summary()}")
                if response.haslayer(TCP) and response[TCP].flags == 0x12:  # SYN-ACK
                    results.update({"status": "open", "confidence": 0.95, "bypass_method": "syn", "protection": "none"})
                    service, confidence = self._predict_service_heuristic(response, response_time)
                    if self.lstm_model:
                        features: torch.Tensor = self._extract_features(response, response_time)
                        lstm_service, lstm_confidence = self._predict_service_lstm(features)
                        if lstm_confidence > confidence:
                            service, confidence = lstm_service, lstm_confidence
                    results.update({"service": service, "confidence": confidence})
                    os_guess: str = await self.fingerprint_tcp_ip(response)
                    results["os_guess"] = os_guess
                    logger.info(f"Port {ip}:{port} ouvert via SYN scan : {service}")
                elif response.haslayer(TCP) and (response[TCP].flags == 0x14 or response[TCP].flags == 0x04):  # RST ou RST-ACK
                    results["status"] = "closed"
                    logger.debug(f"Port {ip}:{port} fermé (RST reçu)")
                elif response.haslayer(ICMP) and response[ICMP].type == 3: # ICMP Destination Unreachable
                    results.update({"status": "filtered", "protection": "firewall", "confidence": 0.7})
                    logger.debug(f"Port {ip}:{port} filtré (ICMP type 3 reçu)")
                else: # Autre réponse TCP ou non-TCP
                    results["status"] = "filtered" # Ou un autre statut selon la réponse
                    logger.debug(f"Port {ip}:{port} filtré (réponse inattendue: {response.summary()})")
            else: # Aucune réponse
                results["status"] = "filtered" # Ou "open|filtered"
                self.packet_loss_count += 1
                logger.debug(f"Port {ip}:{port} : aucune réponse SYN (probablement filtré ou perte de paquet)")
            return results
        except Exception as e: # Inclut les erreurs potentielles de Scapy si les privilèges sont insuffisants
            logger.error(f"Erreur lors du scan SYN pour {ip}:{port} : {e}", exc_info=True)
            self.packet_loss_count += 1
            results.update({"status": "error", "service": f"Erreur Scapy: {e}"}) # Indiquer une erreur
            return results

    async def grab_banner(self, ip: str, port: int, domain: Optional[str]) -> Dict[str, Any]:
        """
        Tente de récupérer une bannière d'un service ouvert.

        Se connecte au port et lit les premiers octets reçus, qui constituent souvent une bannière
        d'identification du service (ex: "SSH-2.0-OpenSSH_8.2p1", "HTTP/1.1 200 OK").

        Args:
            ip (str): L'adresse IP cible.
            port (int): Le port cible.
            domain (Optional[str]): Le nom de domaine (utilisé pour logging/contexte, non essentiel ici).

        Returns:
            Dict[str, Any]: Un dictionnaire contenant:
                - "banner" (str): La bannière brute décodée en UTF-8 (ou vide).
                - "service" (str): Service inféré de la bannière ("http", "ssh", ou "unknown").
                - "inferred_service" (str): Identique à "service".
                - "confidence" (float): Confiance de l'inférence de service.
                - "version" (str): Version (généralement vide à ce stade).
        """
        async with self.semaphore: # Contrôle de concurrence
            results: Dict[str, Any] = {"banner": "", "service": "unknown", "inferred_service": "unknown", "confidence": 0.5, "version": ""}
        try:
            # Tenter une connexion TCP standard pour lire la bannière
            async with asyncio.timeout(self.config.TIMEOUT):
                reader, writer = await asyncio.open_connection(ip, port) # type: ignore
                banner_bytes: bytes = await reader.read(1024) # Lire jusqu'à 1024 octets pour la bannière
                writer.close() # Fermer la connexion
                await writer.wait_closed()

                # Décoder la bannière en UTF-8, ignorant les erreurs de décodage
                banner_str: str = banner_bytes.decode("utf-8", errors="ignore").strip()
                results["banner"] = banner_str

                # Inférence basique du service à partir de mots-clés dans la bannière
                if "HTTP" in banner_str.upper(): # Vérification insensible à la casse
                    results.update({"service": "http", "inferred_service": "http", "confidence": 0.9})
                elif "SSH" in banner_str.upper():
                    results.update({"service": "ssh", "inferred_service": "ssh", "confidence": 0.9})
                # D'autres services (FTP, SMTP, etc.) pourraient être ajoutés ici avec leurs mots-clés respectifs

                logger.debug(f"Bannière récupérée pour {ip}:{port} : {banner_str[:50]}...") # Log tronqué de la bannière
                await asyncio.sleep(random.uniform(0.1, 0.5)) # Petit délai aléatoire
        except Exception as e: # Gérer les erreurs (timeout, connexion refusée, etc.)
            logger.debug(f"Échec de la récupération de la bannière pour {ip}:{port} : {e}")
            self.packet_loss_count += 1
        return results

    async def identify_service_version(self, ip: str, port: int, banner: str) -> Dict[str, Any]:
        """
        Identifie la version d'un service à partir de sa bannière en utilisant des expressions régulières.

        Args:
            ip (str): L'adresse IP (pour logging).
            port (int): Le port (pour logging).
            banner (str): La bannière du service.

        Returns:
            Dict[str, Any]: Un dictionnaire contenant:
                - "version" (str): La version extraite (ou vide si non trouvée).
                - "confidence" (float): Confiance de l'extraction (0.95 si trouvée, 0.7 sinon).
        """
        async with self.semaphore: # Contrôle de concurrence
            results: Dict[str, Any] = {"version": "", "confidence": 0.7} # Initialisation des résultats

        # Liste de couples (expression régulière, type de service) pour extraire la version
        version_patterns: List[Tuple[str, str]] = [
            (r"Server: ([^\r\n]+)", "http"), # Pour les serveurs HTTP (ex: Apache, Nginx)
            (r"SSH-[\d.]+-([^\s]+)", "ssh"),   # Pour les serveurs SSH (ex: OpenSSH_8.2p1)
            (r"Apache/([\d.]+)", "apache"),    # Spécifique pour Apache
            (r"nginx/([\d.]+)", "nginx"),      # Spécifique pour Nginx
            (r"Shopify", "shopify")            # Cas spécifique pour Shopify (pas de version numérique typique)
        ]
        try:
            # Itérer sur chaque pattern pour chercher une correspondance dans la bannière
            for pattern_str, service_hint in version_patterns:
                # `re.search` pour trouver la première occurrence, insensible à la casse
                match: Optional[re.Match[str]] = re.search(pattern_str, banner, re.IGNORECASE)
                if match:
                    # Si une correspondance est trouvée, extraire le groupe capturant la version
                    # Pour Shopify, la "version" est juste "Shopify"
                    version_str: str = match.group(1).strip() if service_hint != "shopify" else "Shopify"
                    results.update({"version": version_str, "confidence": 0.95}) # Mettre à jour avec la version et une haute confiance
                    logger.debug(f"Version identifiée pour {ip}:{port} ({service_hint}) : {version_str}")
                    break # Arrêter après la première correspondance réussie
            await asyncio.sleep(random.uniform(0.1, 0.5)) # Petit délai
        except Exception as e: # Gérer les erreurs potentielles lors de l'analyse regex
            logger.error(f"Erreur lors de l'identification de la version pour {ip}:{port} : {e}")
        return results

    async def enumerate_web_paths(self, ip: str, port: int, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Énumère une liste de chemins web courants sur un serveur HTTP/HTTPS.

        Tente de requêter des chemins comme "/admin", "/login", etc., en utilisant
        des proxies et des User-Agents variés.

        Args:
            ip (str): L'adresse IP du serveur web.
            port (int): Le port du serveur web (ex: 80, 443).
            domain (Optional[str]): Le nom de domaine associé (pour l'en-tête Host et la base de l'URL).

        Returns:
            List[Dict[str, Any]]: Une liste de dictionnaires, chaque dictionnaire représentant un chemin trouvé
                                  et contenant "path", "status_code", et "confidence".
                                  Peut aussi inclure "redirect_to" pour les redirections.
        """
        async with self.semaphore: # Contrôle de concurrence pour le bloc de code externe
            # Liste des chemins sensibles ou courants à vérifier
            sensitive_paths: List[str] = [
                "/admin", "/login", "/cart", "/checkout", "/.env", "/config", "/api",
                "/wp-admin", "/administrator", "/robots.txt", "/sitemap.xml" # Ajout de quelques chemins courants
            ]
        results: List[Dict[str, Any]] = []

        # Déterminer le schéma (http/https) et l'hôte cible pour construire l'URL de base
        scheme: str = "https" if port == 443 or port == 8443 else "http"
        target_host: str = domain or ip # Utiliser le domaine si fourni, sinon l'IP
        target_base: str = f"{scheme}://{target_host}:{port}"

        try:
            # Un timeout global pour l'ensemble de l'opération d'énumération des chemins pour cette cible
            # Multiplier le timeout de base par le nombre de chemins pour donner une marge raisonnable
            # Le `* 2` est une heuristique, pourrait être ajusté.
            async with asyncio.timeout(self.config.TIMEOUT * 2 * len(sensitive_paths)):
                # Préparer les en-têtes HTTP communs pour les requêtes
                headers: Dict[str, str] = {
                    "User-Agent": random.choice(self.config.USER_AGENTS), # Rotation de User-Agent
                    "Host": target_host,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Connection": "keep-alive", # Maintenir la connexion pour plusieurs requêtes si possible
                    "X-Forwarded-For": f"192.168.{random.randint(0, 255)}.{random.randint(1, 255)}" # Tentative de spoofing d'IP interne
                }

                # Créer un contexte SSL pour les requêtes HTTPS, en utilisant certifi pour les CA certificats
                ssl_context_aiohttp: Optional[ssl.SSLContext] = ssl.create_default_context(cafile=certifi.where()) if scheme == "https" else None

                # Utiliser une seule session aiohttp.ClientSession pour toutes les requêtes vers cet hôte
                # Cela permet de réutiliser les connexions (pooling) et d'améliorer les performances.
                # `limit_per_host` contrôle le nombre de connexions simultanées vers le même hôte.
                async with aiohttp.ClientSession(headers=headers, connector=aiohttp.TCPConnector(ssl=ssl_context_aiohttp, limit_per_host=10)) as session:
                    tasks: List[asyncio.Task] = [] # Liste pour stocker les tâches asyncio
                    # Pour chaque chemin sensible, créer une tâche pour le récupérer
                    for path_to_check in sensitive_paths:
                        current_proxy: Optional[Dict[str, str]] = self._get_next_proxy() # Obtenir le prochain proxy pour la rotation
                        # aiohttp attend l'URL complète du proxy (ex: "http://user:pass@host:port")
                        proxy_url_for_request: Optional[str] = current_proxy["http"] if current_proxy and "http" in current_proxy else None

                        # Créer une tâche pour la fonction _fetch_web_path
                        task = asyncio.create_task(self._fetch_web_path(session, f"{target_base}{path_to_check}", path_to_check, ip, port, proxy_url_for_request))
                        tasks.append(task)

                    # Attendre que toutes les tâches de récupération de chemin se terminent
                    # `return_exceptions=True` pour que les exceptions soient retournées comme résultats plutôt que de lever immédiatement
                    path_fetch_results = await asyncio.gather(*tasks, return_exceptions=True)

                    # Traiter les résultats de chaque tâche
                    for res_item in path_fetch_results:
                        if isinstance(res_item, dict) and res_item: # Si le résultat est un dictionnaire valide
                            results.append(res_item)
                        elif isinstance(res_item, Exception): # Si une exception a été retournée par une tâche
                            logger.error(f"Exception lors du fetch d'un chemin web pour {ip}:{port} : {res_item}", exc_info=res_item)
                            self.packet_loss_count +=1 # Compter comme une perte ou une erreur

        except asyncio.TimeoutError: # Gérer le timeout global pour l'énumération
            logger.warning(f"Timeout global lors de l'énumération des chemins pour {ip}:{port}.")
            self.packet_loss_count += 1
        except Exception as e: # Gérer toute autre erreur majeure
            logger.error(f"Erreur majeure lors de l'énumération des chemins pour {ip}:{port} : {e}", exc_info=True)
            self.packet_loss_count += 1
        return results

    async def _fetch_web_path(self, session: aiohttp.ClientSession, url: str, path: str, ip: str, port: int, proxy_url: Optional[str]) -> Optional[Dict[str, Any]]:
        """
        Récupère un chemin web spécifique en utilisant une session aiohttp.

        Args:
            session (aiohttp.ClientSession): La session client HTTP à utiliser.
            url (str): L'URL complète à requêter.
            path (str): Le chemin relatif (pour le rapport).
            ip (str): L'IP (pour logging).
            port (int): Le port (pour logging).
            proxy_url (Optional[str]): L'URL du proxy à utiliser (ex: "http://user:pass@host:port").

        Returns:
            Optional[Dict[str, Any]]: Un dictionnaire avec "path", "status_code", "confidence"
                                      et potentiellement "redirect_to", ou None en cas d'erreur.
        """
        try:
            # Effectuer une requête GET vers l'URL spécifiée.
            # `allow_redirects=False` pour traiter les redirections manuellement si nécessaire.
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=self.config.TIMEOUT), proxy=proxy_url, allow_redirects=False) as response:
                status: int = response.status # Obtenir le code de statut HTTP

                # Préparer les données de résultat de base
                result_data: Dict[str, Any] = {"path": path, "status_code": status}

                # Interpréter différents codes de statut
                if status == 200: # OK
                    logger.debug(f"Chemin web {url} accessible (status 200)")
                    result_data.update({"confidence": 0.95})
                elif 300 <= status < 400 : # Codes de redirection (301, 302, 307, 308)
                    redirect_location = response.headers.get('Location', '')
                    logger.debug(f"Chemin web {url} redirigé (status {status}) vers {redirect_location}")
                    result_data.update({"confidence": 0.9, "redirect_to": redirect_location})
                elif status == 403: # Forbidden
                    logger.debug(f"Chemin web {url} bloqué (status 403, possible WAF/ACL)")
                    result_data.update({"confidence": 0.85, "protection_hint": "WAF/ACL"}) # Indice de protection
                elif status == 401: # Unauthorized
                    logger.debug(f"Chemin web {url} nécessite une authentification (status 401)")
                    result_data.update({"confidence": 0.85, "auth_required": True}) # Indique que l'authentification est requise
                elif status == 404: # Not Found
                    logger.debug(f"Chemin web {url} non trouvé (status 404)")
                    return None # Ne pas rapporter les 404 par défaut, sauf si explicitement souhaité
                else: # Autres codes de statut
                    logger.debug(f"Chemin web {url} a retourné un statut inattendu: {status}")
                    result_data.update({"confidence": 0.7}) # Confiance plus faible pour les statuts inattendus
                return result_data
        except asyncio.TimeoutError: # Gérer les timeouts de requête
            logger.debug(f"Timeout lors de la récupération du chemin {url} avec proxy {proxy_url}")
            self.packet_loss_count += 1
        except aiohttp.ClientError as e: # Gérer les erreurs spécifiques à aiohttp (ex: erreur de connexion)
            logger.debug(f"Erreur client aiohttp pour {url} avec proxy {proxy_url}: {type(e).__name__} - {e}")
            self.packet_loss_count += 1
        except Exception as e: # Gérer toute autre exception
            logger.error(f"Erreur inattendue lors de la récupération de {url} avec proxy {proxy_url}: {e}", exc_info=True)
            self.packet_loss_count += 1
        return None # Retourner None en cas d'erreur ou si le chemin n'est pas jugé pertinent (ex: 404)

    async def advanced_ssl_analysis(self, ip: str, port: int, domain: Optional[str] = None) -> Dict[str, Any]:
        """
        Effectue une analyse SSL/TLS avancée sur un port (généralement 443).

        Récupère les détails du certificat SSL, tels que l'émetteur, les domaines SAN,
        la date d'expiration, le numéro de série, les protocoles et les chiffrements.
        Utilise une exécution synchrone dans un thread pour éviter de bloquer asyncio.

        Args:
            ip (str): L'adresse IP cible.
            port (int): Le port SSL/TLS cible.
            domain (Optional[str]): Le nom de domaine (pour SNI et vérification du nom d'hôte).

        Returns:
            Dict[str, Any]: Un dictionnaire contenant les détails du certificat SSL.
                            Inclut "error" si une erreur s'est produite.
        """
        async with self.semaphore: # Contrôle de concurrence
            results: Dict[str, Any] = { # Initialisation du dictionnaire de résultats
                "issuer": "", "san_domains": [], "not_after": "", "serial_number": "",
                "confidence": 0.0, "error": "", "protocols": [], "cipher": ""
            }

        # Le `hostname` est crucial pour SNI (Server Name Indication) lors de la négociation SSL/TLS.
        # Si un domaine est fourni et différent de l'IP, l'utiliser pour SNI. Sinon, None (ou l'IP si le serveur le supporte).
        hostname_for_sni: Optional[str] = domain if domain and domain != ip else None

        try:
            # L'analyse SSL via `socket` et `ssl` est bloquante.
            # Elle est donc exécutée dans un thread séparé via `run_in_executor` pour ne pas bloquer la boucle asyncio.
            # `self.state.executor` doit être un `concurrent.futures.Executor` (typiquement ThreadPoolExecutor).
            if self.state.executor is None: # Vérification que l'executor est disponible
                logger.error("Executor non initialisé pour advanced_ssl_analysis.")
                results["error"] = "Executor non initialisé (requis pour les opérations bloquantes SSL)"
                return results

            ssl_details: Dict[str, Any] = await asyncio.get_event_loop().run_in_executor(
                self.state.executor, # L'exécuteur de threads
                self._sync_advanced_ssl_analysis, # La fonction synchrone à exécuter
                ip, port, hostname_for_sni # Les arguments pour la fonction synchrone
            )
            results.update(ssl_details) # Mettre à jour les résultats avec les détails SSL obtenus
            await asyncio.sleep(random.uniform(0.1, 0.5)) # Petit délai pour éviter une surcharge potentielle

        except Exception as e: # Gérer les erreurs au niveau de l'orchestration asyncio
            logger.error(f"Erreur majeure lors de l'orchestration de l'analyse SSL pour {ip}:{port} : {e}", exc_info=True)
            results["error"] = f"Erreur d'orchestration SSL: {str(e)}"
            self.packet_loss_count += 1 # Compter comme une perte/erreur
        return results

    def _sync_advanced_ssl_analysis(self, ip: str, port: int, hostname: Optional[str]) -> Dict[str, Any]:
        """
        Fonction synchrone pour l'analyse SSL. Destinée à être exécutée dans un thread.

        Args:
            ip (str): L'adresse IP cible.
            port (int): Le port SSL/TLS cible.
            hostname (Optional[str]): Le nom d'hôte pour SNI.

        Returns:
            Dict[str, Any]: Dictionnaire des détails SSL.
        """
        results: Dict[str, Any] = {"issuer": "", "san_domains": [], "not_after": "", "serial_number": "", "confidence": 0.0, "error": "", "protocols": [], "cipher": ""}
    context = ssl.create_default_context(cafile=certifi.where())
    context.check_hostname = False
    context.verify_mode = ssl.CERT_OPTIONAL
    try:
        with socket.create_connection((ip, port), timeout=self.config.TIMEOUT) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                # getpeercert() peut retourner None ou un Dict.
                # Spécifier le type pour cert_dict pour clarifier.
                cert_dict: Optional[Dict[str, Any]] = ssock.getpeercert()
                if cert_dict:
                    issuer_tuples: List[Tuple[Tuple[str, str], ...]] = cert_dict.get("issuer", [])
                    issuer_info_list: List[Tuple[str,str]] = [item for sublist in issuer_tuples for item in sublist] # type: ignore
                    issuer_dict: Dict[str, str] = dict(issuer_info_list)

                    san_tuples: List[Tuple[str, str]] = cert_dict.get("subjectAltName", [])

                    # Tentative de récupération des protocoles et du cipher
                    # ssock.version() peut être None
                    protocol_version: Optional[str] = ssock.version()
                    # ssock.cipher() peut être None ou un tuple (str, str, int)
                    cipher_info: Optional[Tuple[str,str,int]] = ssock.cipher()

                    results.update({
                        "issuer": issuer_dict.get("commonName", issuer_dict.get("organizationName", "Unknown")),
                        "san_domains": [val for type_val, val in san_tuples if type_val.lower() == 'dns'],
                        "not_after": cert_dict.get("notAfter", ""),
                        "serial_number": cert_dict.get("serialNumber", ""),
                        "protocols": [protocol_version] if protocol_version else [],
                        "cipher": cipher_info[0] if cipher_info else "",
                        "confidence": 0.95
                    })
                    logger.debug(f"Analyse SSL réussie pour {ip}:{port}")
                else:
                    results.update({"error": "getpeercert() returned None", "confidence": 0.4})
                    logger.warning(f"getpeercert() a retourné None pour {ip}:{port}")

    except socket.timeout:
        results.update({"error": "Socket timeout", "confidence": 0.5})
        logger.warning(f"Timeout SSL pour {ip}:{port}")
        self.packet_loss_count += 1 # Considéré comme une forme de perte/échec
    except ssl.SSLError as e: # Erreurs SSL spécifiques
        results.update({"error": f"SSLError: {e.reason}", "confidence": 0.5})
        logger.warning(f"Erreur SSL pour {ip}:{port} : {e.reason}", exc_info=False) # exc_info=False pour éviter trop de logs
        self.packet_loss_count += 1
    except Exception as e: # Autres erreurs (ex: ConnectionRefusedError)
        results.update({"error": f"Generic error: {str(e)}", "confidence": 0.5})
        logger.error(f"Erreur générique lors de l'analyse SSL pour {ip}:{port} : {e}", exc_info=True)
        self.packet_loss_count += 1
    return results

    async def check_vulnerabilities(self, ip: str, port: int, version: str, banner: str) -> List[Dict[str, Any]]:
        """
        Vérifie les vulnérabilités connues basées sur la version du service et la bannière.

        Utilise une base de connaissances interne (dictionnaire `known_vulnerabilities`)
        pour faire correspondre les versions/bannières à des CVEs ou des faiblesses connues.

        Args:
            ip (str): L'adresse IP (pour logging).
            port (int): Le port (pour logging).
            version (str): La version du service détectée.
            banner (str): La bannière du service.

        Returns:
            List[Dict[str, Any]]: Une liste de dictionnaires, chaque dictionnaire
                                  représentant une vulnérabilité trouvée et contenant
                                  "description", "severity", et "confidence".
        """
        async with self.semaphore: # Contrôle de concurrence
            results: List[Dict[str, Any]] = [] # Initialiser la liste des résultats de vulnérabilités

        # Base de connaissances simplifiée des vulnérabilités.
        # Idéalement, les expressions régulières seraient compilées une seule fois lors de l'initialisation de la classe
        # pour de meilleures performances si cette méthode est appelée très fréquemment.
        known_vulnerabilities: Dict[str, List[Tuple[re.Pattern, str, str]]] = {
            "apache": [
                (re.compile(r"^2\.2\..*"), "Apache 2.2.x obsolète (multiples CVEs)", "critical"), # Vulnérabilité pour Apache 2.2.x
                (re.compile(r"^2\.4\.([0-9]|[1-4][0-9]|5[0-3])$"), "Apache 2.4.x < 2.4.54 (vérifiez CVE-2021-42013, CVE-2022-31813 etc.)", "high"), # Vulnérabilité pour Apache 2.4.x avant une certaine version patch
            ],
            "nginx": [
                (re.compile(r"^1\.(1[0-9]|2[0-1])\..*"), "Nginx < 1.22.0 (multiples CVEs, ex: CVE-2021-23017)", "medium"), # Vulnérabilité pour Nginx avant 1.22.0
            ],
            "shopify": [ # Pour Shopify, il s'agit plus de recommandations de configuration que de CVEs directes
                (re.compile(r".*"), "Vérifiez les configurations WAF (ex: Cloudflare) et les apps tierces installées.", "medium")
            ]
        }
        try:
            identified_service: str = "" # Service identifié pour la recherche de vulnérabilités

            # Normaliser la version et la bannière en minuscules pour une correspondance insensible à la casse
            version_lower: str = version.lower()
            banner_lower: str = banner.lower()

            # Déterminer le type de service basé sur la version ou la bannière
            if "apache" in version_lower or "apache" in banner_lower:
                identified_service = "apache"
            elif "nginx" in version_lower or "nginx" in banner_lower:
                identified_service = "nginx"
            elif "shopify" in banner_lower: # Shopify est souvent identifié par sa présence dans la bannière
                identified_service = "shopify"

            # Si un service a été identifié et qu'il existe dans la base de connaissances
            if identified_service in known_vulnerabilities:
                # Parcourir les patterns de vulnérabilité pour ce service
                for pattern, desc, severity in known_vulnerabilities[identified_service]:
                    # Vérifier si le pattern correspond à la version ou à la bannière
                    if pattern.search(version) or pattern.search(banner):
                        results.append({"description": desc, "severity": severity, "confidence": 0.95, "source_match": "version/banner"})
                        logger.debug(f"Vulnérabilité potentielle détectée pour {ip}:{port} ({identified_service}) : {desc}")

            await asyncio.sleep(random.uniform(0.1, 0.5)) # Petit délai
        except Exception as e: # Gérer les erreurs potentielles
            logger.error(f"Erreur lors de la vérification des vulnérabilités pour {ip}:{port} : {e}", exc_info=True)
        return results

    async def advanced_subdomain_enumeration(self, ip: str, domain: Optional[str]) -> List[str]:
        """
        Énumère les sous-domaines courants pour un domaine donné et vérifie s'ils résolvent vers l'IP cible.

        Args:
            ip (str): L'adresse IP cible à laquelle les sous-domaines doivent correspondre.
            domain (Optional[str]): Le domaine de base pour l'énumération (ex: "shopify.com").

        Returns:
            List[str]: Une liste de sous-domaines valides qui résolvent vers l'IP cible.
                       Retourne une liste vide si aucun domaine n'est fourni, si le domaine est une IP,
                       ou en cas d'erreur.
        """
        async with self.semaphore: # Contrôle de concurrence
            subdomains_found: List[str] = [] # Liste pour stocker les sous-domaines valides trouvés

        # Vérifier si un domaine valide est fourni (non nul et pas une adresse IP)
        if not domain or re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", domain):
            logger.debug(f"Énumération des sous-domaines ignorée pour IP {ip} / Domaine {domain} : aucun domaine valide fourni ou domaine est une IP.")
            return subdomains_found # Retourner une liste vide si pas de domaine valide

        # Liste de préfixes de sous-domaines courants à tester
        common_subdomains_list: List[str] = ["www", "mail", "api", "admin", "test", "shop", "store", "dev", "staging", "cdn", "assets"]

        tasks: List[asyncio.Task] = [] # Liste pour stocker les tâches de vérification de sous-domaine

        # Appliquer un timeout global pour l'ensemble de l'opération d'énumération des sous-domaines
        # Le timeout est basé sur le nombre de sous-domaines à vérifier et le timeout de configuration.
        # Le `/ 2` est une heuristique pour donner plus de temps que pour une seule requête DNS.
        try:
            async with asyncio.timeout(self.config.TIMEOUT * (len(common_subdomains_list) / 2)):
                # Créer une tâche pour chaque sous-domaine à vérifier
                for sub_prefix in common_subdomains_list:
                    full_subdomain_to_check = f"{sub_prefix}.{domain}"
                    task = asyncio.create_task(self._check_subdomain_resolves_to_ip(full_subdomain_to_check, ip))
                    tasks.append(task)

                # Attendre que toutes les tâches de vérification se terminent
                # `return_exceptions=True` pour capturer les exceptions de résolution DNS individuellement
                resolution_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Traiter les résultats de chaque tâche
                for res_item in resolution_results:
                    if isinstance(res_item, str): # Si la tâche a retourné un nom de sous-domaine (succès)
                        subdomains_found.append(res_item)
                        logger.debug(f"Sous-domaine {res_item} résout vers {ip} pour le domaine {domain}")
                    elif isinstance(res_item, Exception): # Si une exception s'est produite pendant la résolution
                        logger.debug(f"Exception lors de la vérification du sous-domaine pour {domain}: {res_item}")
                        # Les exceptions attendues comme NXDOMAIN ne sont pas comptées comme des pertes de paquets.

        except asyncio.TimeoutError: # Gérer le timeout global
            logger.warning(f"Timeout global lors de l'énumération des sous-domaines pour {domain} (IP: {ip}).")
            self.packet_loss_count +=1 # Compter comme une perte/erreur
        except Exception as e: # Gérer toute autre erreur majeure
            logger.error(f"Erreur majeure lors de l'énumération des sous-domaines pour {domain} (IP: {ip}): {e}", exc_info=True)
            self.packet_loss_count += 1

        return list(set(subdomains_found)) # Retourner une liste unique de sous-domaines trouvés

    async def _check_subdomain_resolves_to_ip(self, subdomain: str, target_ip: str) -> Optional[str]:
        """
        Vérifie si un sous-domaine donné résout (enregistrement A) vers une adresse IP spécifique.

        Args:
            subdomain (str): Le sous-domaine complet à vérifier (ex: "www.shopify.com").
            target_ip (str): L'adresse IP attendue.

        Returns:
            Optional[str]: Le sous-domaine s'il résout vers `target_ip`, sinon None.
        """
        try:
            # Exécuter la résolution DNS (opération bloquante) dans un thread séparé
            answers: Any = await asyncio.get_event_loop().run_in_executor(
                None,  # Utilise l'exécuteur de threads par défaut d'asyncio
                lambda: dns.resolver.resolve(subdomain, 'A') # Résoudre les enregistrements A (adresses IPv4)
            )
            # Parcourir les adresses IP résolues
            for rdata in answers:
                if rdata.address == target_ip: # Si l'une des adresses correspond à l'IP cible
                    logger.debug(f"Confirmation: {subdomain} résout vers {target_ip}")
                    return subdomain # Retourner le sous-domaine
            # Si le sous-domaine résout vers d'autres IPs, mais pas celle ciblée
            logger.debug(f"Le sous-domaine {subdomain} résout vers d'autres IPs ({[r.address for r in answers]}) que {target_ip}.")
        except dns.resolver.NXDOMAIN: # Non-Existent Domain
            logger.debug(f"Le sous-domaine {subdomain} n'existe pas (NXDOMAIN).")
        except dns.resolver.NoAnswer: # Le domaine existe, mais pas d'enregistrement du type demandé (A)
            logger.debug(f"Aucune réponse A pour le sous-domaine {subdomain}.")
        except dns.resolver.Timeout: # Timeout lors de la requête DNS
            logger.debug(f"Timeout lors de la résolution DNS pour {subdomain}.")
            self.packet_loss_count +=1 # Peut indiquer un problème réseau ou un serveur DNS lent
        except Exception as e: # Capturer d'autres exceptions de `dns.resolver` (SERVFAIL, etc.)
            logger.debug(f"Erreur de résolution DNS pour {subdomain} : {type(e).__name__} - {e}")

        # Pas de délai aléatoire ici, car cette fonction est une aide appelée potentiellement en boucle.
        # La gestion des délais est faite par la fonction appelante.
        return None # Retourner None si le sous-domaine ne résout pas vers l'IP cible ou en cas d'erreur

    async def waf_bypass(self, ip: str, port: int, domain: Optional[str] = None) -> Dict[str, Any]:
        """
        Tente diverses techniques pour contourner un Web Application Firewall (WAF).

        Utilise une rotation de proxies, des User-Agents variés, et des en-têtes HTTP
        spécifiques pour essayer d'accéder à la ressource cible.

        Args:
            ip (str): L'adresse IP cible.
            port (int): Le port cible (généralement 80 ou 443).
            domain (Optional[str]): Le nom de domaine (pour l'en-tête Host et l'URL).

        Returns:
            Dict[str, Any]: Un dictionnaire indiquant si le bypass a réussi,
                            le WAF détecté (si possible), la confiance et la méthode utilisée.
        """
        async with self.semaphore:
            results: Dict[str, Any] = {"bypass_success": False, "waf_detected": "none", "confidence": 0.0, "bypass_method": "none"}
        target_url = f"http://{domain or ip}:{port}" if port == 80 else f"https://{domain or ip}:{port}"
        payloads = [
            {
                "name": "Basic GET with X-Forwarded-For",
                "headers": { # type: ignore
                    "User-Agent": random.choice(self.config.USER_AGENTS),
                    "X-Forwarded-For": f"192.168.{random.randint(0, 255)}.{random.randint(1, 255)}", # IP interne commune
                    "Host": domain or ip,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Connection": "keep-alive"
                },
                "url_param": "?bypass=true&rand=" + str(random.randint(1000,9999)), # Paramètre de contournement commun
                "method": "GET"
            },
            {
                "name": "HEAD with Googlebot User-Agent",
                "headers": { # type: ignore
                    "User-Agent": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
                    "Host": domain or ip,
                    "Accept": "text/html",
                    "Connection": "keep-alive"
                },
                "url_param": "",
                "method": "HEAD" # Moins susceptible d'être bloqué par certains WAFs
            },
            {
                "name": "GET with Referer and Origin",
                "headers": { # type: ignore
                    "User-Agent": random.choice(self.config.USER_AGENTS),
                    "Host": domain or ip,
                    "Origin": f"https://{domain or ip}", # Simule une requête same-origin
                    "Referer": f"https://{domain or ip}/somepage.html",
                     "X-Forwarded-For": f"{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}" # IP publique aléatoire
                },
                "url_param": "/?rand=" + str(random.randint(1000, 9999)),
                "method": "GET"
            }
        ]

        # Déterminer le schéma pour le contexte SSL
        scheme: str = "https" if port == 443 or port == 8443 else "http"
        ssl_context_aiohttp: Optional[ssl.SSLContext] = ssl.create_default_context(cafile=certifi.where()) if scheme == "https" else None

        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context_aiohttp, limit_per_host=5)) as session:
            for attempt_num, payload_info in enumerate(payloads):
                current_headers: Dict[str, str] = payload_info["headers"] # type: ignore
                current_url: str = target_url + payload_info.get("url_param", "")
                method: str = payload_info["method"] # type: ignore
                current_proxy: Optional[Dict[str, str]] = self._get_next_proxy()
                proxy_url: Optional[str] = current_proxy.get("http") if current_proxy else None # Utiliser la clé "http" pour aiohttp

                logger.debug(f"WAF bypass: Tentative '{payload_info['name']}' ({attempt_num+1}/{len(payloads)}) pour {current_url} via proxy {proxy_url}")

                try:
                    async with asyncio.timeout(self.config.TIMEOUT): # Timeout par tentative
                        async with session.request(method, current_url, headers=current_headers, proxy=proxy_url, allow_redirects=False) as response:
                            status: int = response.status
                            # Lire le corps pour l'analyse, même si c'est une requête HEAD (le corps sera vide)
                            response_text: str = await response.text(errors='ignore')

                            # Détection de WAF basique
                            server_header: str = response.headers.get("Server", "").lower()
                            waf_keywords: List[str] = ["cloudflare", "akamai", "sucuri", "incapsula", "barracuda", "f5", "mod_security", "awswaf", "shieldsquare", "imperva"]
                            response_content_lower: str = response_text.lower()
                            headers_lower: str = str(response.headers).lower()

                            detected_waf_name: str = "none"
                            for kw in waf_keywords:
                                if kw in server_header or kw in response_content_lower or kw in headers_lower:
                                    detected_waf_name = kw
                                    break

                            results["waf_detected"] = detected_waf_name # Mettre à jour le WAF détecté à chaque tentative

                            if 200 <= status < 300: # Succès si status 2xx
                                results.update({
                                    "bypass_success": True,
                                    "confidence": 0.95,
                                    "waf_detected": detected_waf_name if detected_waf_name != "none" else "bypassed_unknown_waf",
                                    "bypass_method": payload_info["name"] # type: ignore
                                })
                                logger.info(f"WAF bypass: Succès pour {current_url} (status {status}) avec méthode '{payload_info['name']}'. WAF: {results['waf_detected']}")
                                return results # Sortir dès le premier succès
                            elif status in [403, 406, 429, 500, 503]: # Codes d'erreur typiques de WAF ou de serveur surchargé
                                results.update({"confidence": 0.85, "bypass_success": False})
                                logger.info(f"WAF bypass: Bloqué pour {current_url} (status {status}) avec méthode '{payload_info['name']}'. WAF: {results['waf_detected']}")
                            else: # Autres status
                                results.update({"confidence": 0.80, "bypass_success": False})
                                logger.debug(f"WAF bypass: Status {status} pour {current_url} avec méthode '{payload_info['name']}'. WAF: {results['waf_detected']}")

                except asyncio.TimeoutError:
                    logger.debug(f"WAF bypass: Timeout pour {current_url} (méthode '{payload_info['name']}')")
                    self.packet_loss_count += 1
                except aiohttp.ClientError as e:
                    logger.debug(f"WAF bypass: ClientError pour {current_url} (méthode '{payload_info['name']}') : {type(e).__name__} - {e}")
                    self.packet_loss_count += 1
                except Exception as e:
                    logger.error(f"WAF bypass: Erreur inattendue pour {current_url} (méthode '{payload_info['name']}') : {e}", exc_info=True)
                    self.packet_loss_count += 1

                if attempt_num < len(payloads) - 1: # Ne pas attendre après la dernière tentative
                    await asyncio.sleep(random.uniform(0.2, 0.7)) # Délai un peu plus long entre les tentatives de bypass WAF

        logger.info(f"WAF bypass: Toutes les {len(payloads)} tentatives ont échoué pour {target_url}. Dernier WAF détecté: {results['waf_detected']}")
        return results

    async def cloudflare_stealth_probe(self, ip: str, port: int, domain: Optional[str] = None) -> Dict[str, Any]:
        """
        Effectue une sonde "furtive" pour détecter et potentiellement contourner Cloudflare.

        Utilise des User-Agents communs (y compris ShopifyBot), des en-têtes HTTP spécifiques
        et une rotation de proxy pour tenter d'obtenir une réponse non bloquée.

        Args:
            ip (str): L'adresse IP cible.
            port (int): Le port cible (généralement 80 ou 443).
            domain (Optional[str]): Le nom de domaine pour l'en-tête Host et l'URL.

        Returns:
            Dict[str, Any]: Un dictionnaire contenant les résultats de la sonde, incluant:
                - "service", "version", "banner", "confidence", "bypass_success", "protection".
        """
        async with self.semaphore: # Contrôle de concurrence
            results: Dict[str, Any] = {"service": "unknown", "version": "", "banner": "", "confidence": 0.0, "bypass_success": False, "protection": "none"}

        target_url = f"http://{domain or ip}:{port}" if port == 80 else f"https://{domain or ip}:{port}"
        # Liste d'User-Agents incluant ShopifyBot, qui peut être moins susceptible d'être bloqué par des WAFs configurés pour Shopify
        user_agents = self.config.USER_AGENTS + ["ShopifyBot/1.0 (+https://shopify.com)"]

        # Multiples tentatives avec différents proxies et User-Agents
        for attempt in range(self.config.MAX_RETRIES):
            try:
                # Timeout pour chaque tentative individuelle
                async with asyncio.timeout(self.config.TIMEOUT):
                    # Préparation des en-têtes pour simuler un navigateur légitime
                    headers = {
                        "User-Agent": random.choice(user_agents), # Rotation de User-Agent
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9", # En-tête Accept commun
                        "Accept-Language": "en-US,en;q=0.9,fr-FR;q=0.8,fr;q=0.7", # Préférences de langue
                        "Connection": "keep-alive",
                        "Host": domain or ip, # En-tête Host crucial
                        "Referer": f"https://{domain or ip}/", # Referer commun
                        "DNT": "1", # Do Not Track
                        "Upgrade-Insecure-Requests": "1", # Demander HTTPS si disponible
                        "X-Forwarded-For": f"192.168.{random.randint(0, 255)}.{random.randint(1, 255)}" # Spoofing léger
                    }

                    scheme: str = "https" if port == 443 or port == 8443 else "http"
                    ssl_context_aiohttp: Optional[ssl.SSLContext] = ssl.create_default_context(cafile=certifi.where()) if scheme == "https" else None

                    current_proxy: Optional[Dict[str, str]] = self._get_next_proxy() # Rotation de proxy
                    proxy_url: Optional[str] = current_proxy.get("http") if current_proxy else None

                    # Utiliser une nouvelle session aiohttp pour chaque tentative peut parfois aider à contourner les WAFs basés sur session
                    async with aiohttp.ClientSession(headers=headers, connector=aiohttp.TCPConnector(ssl=ssl_context_aiohttp, limit_per_host=1)) as session:
                        logger.debug(f"Cloudflare stealth probe: Tentative {attempt+1} pour {target_url} avec proxy {proxy_url}, UA: {headers['User-Agent']}")
                        # Effectuer la requête GET, sans suivre les redirections automatiquement pour analyse
                        async with session.get(target_url, proxy=proxy_url, allow_redirects=False) as response:
                            status: int = response.status
                            banner_text: str = await response.text(encoding='utf-8', errors='ignore') # Lire le corps de la réponse

                            results["banner"] = banner_text[:500] # Stocker un extrait de la bannière
                            server_header: str = response.headers.get("Server", "").lower() # En-tête Server

                            # Détection de Cloudflare basée sur les en-têtes et le contenu
                            is_cloudflare_present: bool = (
                                "cloudflare" in server_header or
                                "cf-ray" in response.headers or # En-tête spécifique à Cloudflare
                                "cf-" in str(response.headers).lower() or
                                ("expect-ct" in response.headers and "cloudflare" in response.headers["expect-ct"].lower())
                            )
                            # Détection spécifique de Shopify
                            is_shopify_site: bool = "shopify" in banner_text.lower() or "cdn.shopify.com" in banner_text.lower()

                            if status == 200: # Si la requête est réussie (status 200 OK)
                                results.update({
                                    "service": "https" if scheme == "https" else "http",
                                    "version": "Shopify" if is_shopify_site else (server_header if server_header else "Unknown WebServer"),
                                    "confidence": 0.95,
                                    "bypass_success": True, # La sonde a réussi à accéder directement
                                    "protection": "cloudflare" if is_cloudflare_present else ("shopify_platform" if is_shopify_site else "unknown_or_none")
                                })
                                logger.info(f"Cloudflare stealth probe: Accès direct réussi pour {target_url} (status 200). Protection: {results['protection']}")
                                return results # Succès, retourner immédiatement
                            elif status in [403, 503, 429] or (status == 520 and is_cloudflare_present): # Codes d'erreur typiques de Cloudflare ou WAF
                                results.update({
                                    "protection": "cloudflare" if is_cloudflare_present else "blocked_generic", # Marquer comme bloqué par Cloudflare ou générique
                                    "confidence": 0.90,
                                    "bypass_success": False
                                })
                                logger.info(f"Cloudflare stealth probe: Bloqué pour {target_url} (status {status}). Protection: {results['protection']}")
                                # Continuer les tentatives, car le blocage peut être temporaire ou spécifique au proxy/UA
                            else: # Autres codes de statut inattendus
                                results.update({"confidence": 0.80, "bypass_success": False, "protection": "unknown"})
                                logger.debug(f"Cloudflare stealth probe: Status {status} inattendu pour {target_url}.")

            except asyncio.TimeoutError: # Gérer les timeouts de requête
                logger.debug(f"Cloudflare stealth probe: Timeout pour {target_url} (tentative {attempt+1})")
                self.packet_loss_count += 1
            except aiohttp.ClientError as e: # Gérer les erreurs client aiohttp
                logger.debug(f"Cloudflare stealth probe: ClientError pour {target_url} (tentative {attempt+1}): {type(e).__name__} - {e}")
                self.packet_loss_count += 1
            except Exception as e: # Gérer les autres exceptions
                logger.error(f"Cloudflare stealth probe: Erreur inattendue pour {target_url} (tentative {attempt+1}): {e}", exc_info=True)
                self.packet_loss_count += 1

            if attempt < self.config.MAX_RETRIES - 1: # Attendre avant la prochaine tentative si ce n'est pas la dernière
                await asyncio.sleep(random.uniform(0.3, 0.8))

        # Si toutes les tentatives ont échoué, mettre à jour les résultats finaux
        results.update({"confidence": max(results.get("confidence", 0.0), 0.60), "protection": results.get("protection", "error_or_blocked")})
        logger.info(f"Cloudflare stealth probe: Échec de toutes les tentatives pour {target_url}. Protection finale: {results['protection']}")
        return results

    async def cloudflare_js_challenge_solver(self, ip: str, port: int, domain: Optional[str] = None) -> Dict[str, Any]:
        """
        Tente de résoudre les défis JavaScript de Cloudflare en utilisant Selenium.

        Nécessite `webdriver-manager` et `selenium` installés, ainsi que `chromedriver`
        dans le PATH ou géré par `webdriver-manager`.
        Cette méthode est coûteuse en ressources et en temps.

        Args:
            ip (str): L'adresse IP cible.
            port (int): Le port cible.
            domain (Optional[str]): Le nom de domaine.

        Returns:
            Dict[str, Any]: Un dictionnaire indiquant si le défi a été résolu,
                            la confiance, et la méthode ("js_challenge").
                            Contient "error" si les dépendances sont manquantes ou
                            si une erreur s'est produite.
        """
        async with self.semaphore: # Limiter le nombre d'instances de navigateur simultanées
            results: Dict[str, Any] = {"bypass_success": False, "confidence": 0.0, "bypass_method": "js_challenge"}

        # Importations dynamiques pour Selenium, car ce sont des dépendances optionnelles et lourdes
        try:
            from webdriver_manager.chrome import ChromeDriverManager
            from selenium import webdriver as selenium_webdriver # Alias générique
            from selenium.webdriver.chrome.options import Options as SeleniumOptions # Options spécifiques à Chrome
            from selenium.webdriver.chrome.service import Service as SeleniumService # Service pour ChromeDriver
            from selenium.common.exceptions import TimeoutException as SeleniumTimeoutException # Exception de timeout de Selenium
            from selenium.common.exceptions import WebDriverException as SeleniumWebDriverException # Exception générique de WebDriver
        except ImportError:
            logger.error("JS challenge solver: Les modules 'webdriver-manager' ou 'selenium' ne sont pas installés. Cette fonctionnalité sera désactivée.")
            results.update({"confidence": 0.10, "error": "Dépendances Selenium (webdriver-manager, selenium) non installées."})
            return results # Retourner si les dépendances sont manquantes

        # Construction de l'URL cible
        target_url: str = f"http://{domain or ip}:{port}" if port == 80 else f"https://{domain or ip}:{port}"

        # Vérifier si l'exécuteur (ThreadPoolExecutor) est disponible pour les opérations bloquantes
        if self.state.executor is None:
            logger.error("Executor non initialisé pour cloudflare_js_challenge_solver (requis pour Selenium).")
            results["error"] = "Executor non initialisé"
            return results

        try:
            # Appliquer un timeout global pour toute l'opération Selenium, y compris le démarrage du driver
            # Cela peut être long, surtout la première fois si le driver doit être téléchargé.
            timeout_duration = self.config.TIMEOUT * 15  # Ex: 2s * 15 = 30 secondes. Ajuster au besoin.
            logger.debug(f"JS challenge solver: Démarrage de l'opération Selenium pour {target_url} avec timeout global de {timeout_duration}s.")

            # Exécuter la logique Selenium (bloquante) dans un thread séparé
            selenium_op_results: Dict[str, Any] = await asyncio.get_event_loop().run_in_executor(
                self.state.executor,
                lambda: self._sync_selenium_js_challenge(target_url) # Appel de la méthode synchrone
            )
            results.update(selenium_op_results) # Mettre à jour les résultats avec ceux de l'opération Selenium

        except asyncio.TimeoutError: # Gérer le timeout global pour l'opération Selenium
            logger.warning(f"JS challenge solver: Timeout global ({timeout_duration}s) pour {target_url} (incluant l'installation/démarrage du driver).")
            results.update({"confidence": 0.75, "bypass_success": False, "error": "Timeout global de l'opération Selenium."})
            self.packet_loss_count += 1
        except Exception as e: # Gérer les erreurs inattendues lors de l'appel à l'exécuteur
            logger.error(f"JS challenge solver: Erreur majeure lors de l'exécution de Selenium pour {target_url}: {e}", exc_info=True)
            results.update({"confidence": 0.40, "bypass_success": False, "error": f"Erreur inattendue de l'executor Selenium: {e}"})
        return results

    def _sync_selenium_js_challenge(self, target_url: str) -> Dict[str, Any]:
        """
        Logique synchrone pour la résolution de défi JavaScript avec Selenium.
        Cette méthode est conçue pour être exécutée dans un thread séparé.

        Args:
            target_url (str): L'URL cible à charger dans le navigateur.

        Returns:
            Dict[str, Any]: Résultats de la tentative de résolution du défi.
        """
        results: Dict[str, Any] = {"bypass_success": False, "confidence": 0.0}
    driver = None
    try:
        from webdriver_manager.chrome import ChromeDriverManager
        from selenium.webdriver.chrome.webdriver import WebDriver as ChromeWebDriver # Alias plus spécifique
        from selenium.webdriver.chrome.options import Options as ChromeOptions # Alias
        from selenium.webdriver.chrome.service import Service as ChromeService # Alias
        from selenium.common.exceptions import TimeoutException as SeleniumTimeoutException
        from selenium.common.exceptions import WebDriverException as SeleniumWebDriverException
    except ImportError: # Devrait être capturé par la méthode appelante, mais double-vérification
        logger.critical("Sync JS Challenge: Dépendances Selenium non trouvées (ceci ne devrait pas arriver).")
        return {"bypass_success": False, "confidence": 0.0, "error": "Dépendances Selenium manquantes."}

    webdriver_service = None
    try:
        chrome_options = ChromeOptions()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        # Utiliser un User-Agent spécifique pour Selenium pour le distinguer des requêtes aiohttp
        chrome_options.add_argument(f"--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36 Selenium")

        current_proxy: Optional[Dict[str, str]] = self._get_next_proxy() # Obtenir un proxy pour cette tentative Selenium
        if current_proxy and "http" in current_proxy: # Assurer que la clé http existe
            chrome_options.add_argument(f"--proxy-server={current_proxy['http']}")

        # Gérer l'installation et le service de ChromeDriver
        # Le log_level=logging.WARNING réduit la verbosité de webdriver-manager
        try:
            webdriver_service = ChromeService(ChromeDriverManager(log_level=logging.WARNING).install())
            driver = ChromeWebDriver(service=webdriver_service, options=chrome_options)
        except SeleniumWebDriverException as e_wd_init:
            logger.error(f"Sync JS Challenge: Erreur initialisation WebDriver: {e_wd_init.msg}", exc_info=False) # msg est plus concis
            return {"bypass_success": False, "confidence": 0.2, "error": f"WebDriver init error: {e_wd_init.msg}"}
        except ValueError as e_val_init: # Peut arriver si chromedriver n'est pas compatible
             logger.error(f"Sync JS Challenge: Erreur valeur WebDriver (potentiel problème version chromedriver): {e_val_init}", exc_info=False)
             return {"bypass_success": False, "confidence": 0.2, "error": f"WebDriver value error: {e_val_init}"}


        driver.set_page_load_timeout(self.config.TIMEOUT * 3) # Ex: 2s * 3 = 6s pour chargement de page

        logger.debug(f"Sync JS challenge: Navigation vers {target_url} avec proxy {current_proxy.get('http') if current_proxy else 'None'}")
        driver.get(target_url)

        # Attente initiale pour que la page se charge et que les scripts JS s'exécutent.
        # Cette attente est cruciale pour les pages avec des défis JS.
        time.sleep(self.config.TIMEOUT + 3) # Ex: 2s + 3s = 5s. Ajuster si nécessaire.

        page_title_lower: str = driver.title.lower()
        page_source_snippet_lower: str = driver.page_source[:1000].lower() # Examiner un extrait pour performance

        # Mots-clés communs indiquant un défi JS ou une page d'attente Cloudflare
        challenge_keywords: List[str] = ["just a moment", "checking your browser", "verify you are human", "challenge", "ray id"]

        is_challenge_detected: bool = any(kw in page_title_lower or kw in page_source_snippet_lower for kw in challenge_keywords)

        if is_challenge_detected:
            logger.info(f"Sync JS challenge: Défi JS ou page d'attente détecté pour {target_url}. Attente supplémentaire.")
            time.sleep(self.config.TIMEOUT + 7) # Attente plus longue, ex: 2s + 7s = 9s

            # Ré-évaluer après l'attente
            page_title_lower = driver.title.lower()
            page_source_snippet_lower = driver.page_source[:1000].lower()
            is_challenge_still_present: bool = any(kw in page_title_lower or kw in page_source_snippet_lower for kw in challenge_keywords)

            if is_challenge_still_present:
                results.update({"confidence": 0.85, "bypass_success": False, "error": "Page de défi JS persistante après attente."})
                logger.info(f"Sync JS challenge: Échec pour {target_url}, page de défi persistante.")
            else:
                results.update({"bypass_success": True, "confidence": 0.95})
                logger.info(f"Sync JS challenge: Succès apparent pour {target_url} après attente (défi résolu ou page chargée).")
        else: # Pas de défi détecté initialement
            results.update({"bypass_success": True, "confidence": 0.90}) # Peut-être pas de défi, ou résolu très vite
            logger.info(f"Sync JS challenge: Aucun défi JS détecté initialement pour {target_url} ou résolu rapidement.")

    except SeleniumTimeoutException as e_timeout:
        logger.warning(f"Sync JS challenge: Timeout Selenium ({e_timeout.msg}) pour {target_url}.")
        results.update({"confidence": 0.70, "bypass_success": False, "error": f"Timeout Selenium: {e_timeout.msg}"})
        self.packet_loss_count += 1 # Le timeout peut être dû à des problèmes réseau
    except SeleniumWebDriverException as e_wd: # Autres erreurs WebDriver (ex: navigateur crash)
        logger.error(f"Sync JS challenge: Erreur WebDriver ({e_wd.msg}) pour {target_url}.", exc_info=False)
        results.update({"confidence": 0.30, "bypass_success": False, "error": f"WebDriver error: {e_wd.msg}"})
    except Exception as e_generic: # Erreurs inattendues
        logger.error(f"Sync JS challenge: Erreur inattendue pour {target_url}: {e_generic}", exc_info=True)
        results.update({"confidence": 0.50, "bypass_success": False, "error": f"Erreur inattendue: {e_generic}"})
    finally:
        if driver:
            try:
                driver.quit()
            except Exception as e_quit: # Erreur lors de la fermeture du driver
                logger.warning(f"Sync JS challenge: Erreur lors de la fermeture de WebDriver: {e_quit}", exc_info=False)
        if webdriver_service and hasattr(webdriver_service, 'stop'): # Arrêter le service chromedriver
             try:
                webdriver_service.stop()
             except Exception as e_service_stop:
                logger.warning(f"Sync JS challenge: Erreur lors de l'arrêt du service chromedriver: {e_service_stop}", exc_info=False)

    return results

    async def cloudflare_rate_limit_evasion(self, ip: str, port: int, domain: Optional[str] = None) -> Dict[str, Any]:
        """
        Tente d'éviter les limites de débit de Cloudflare en effectuant des requêtes
        répétées avec rotation de proxy et d'User-Agents.

        Args:
            ip (str): L'adresse IP cible.
            port (int): Le port cible.
            domain (Optional[str]): Le nom de domaine.

        Returns:
            Dict[str, Any]: Un dictionnaire indiquant si l'évasion a réussi
                            ("evasion_success") et la confiance ("confidence").
        """
        async with self.semaphore:
            results: Dict[str, Any] = {"evasion_success": False, "confidence": 0.0}
        target_url = f"http://{domain or ip}:{port}" if port == 80 else f"https://{domain or ip}:{port}"
        user_agents = self.config.USER_AGENTS + ["ShopifyBot/1.0 (+https://shopify.com)"]
        try:
            async with asyncio.timeout(self.config.TIMEOUT * self.config.MAX_RETRIES):
                for attempt in range(self.config.MAX_RETRIES):
                    headers = {
                        "User-Agent": random.choice(user_agents),
                        "X-Forwarded-For": f"192.168.{random.randint(0, 255)}.{random.randint(1, 255)}"
                    }
                    scheme: str = "https" if port == 443 or port == 8443 else "http"
                    ssl_context_aiohttp: Optional[ssl.SSLContext] = ssl.create_default_context(cafile=certifi.where()) if scheme == "https" else None
                    current_proxy: Optional[Dict[str, str]] = self._get_next_proxy()
                    proxy_url: Optional[str] = current_proxy.get("http") if current_proxy else None

                    # Utiliser une nouvelle session pour chaque tentative peut aider
                    async with aiohttp.ClientSession(headers=headers, connector=aiohttp.TCPConnector(ssl=ssl_context_aiohttp, limit_per_host=1)) as session:
                        try:
                            # Pas besoin d'un long délai ici, car on teste la réponse rapide du serveur
                            logger.debug(f"Cloudflare rate limit evasion: Tentative {attempt+1} pour {target_url} avec proxy {proxy_url}, UA: {headers['User-Agent']}")
                            async with session.get(target_url, proxy=proxy_url, timeout=aiohttp.ClientTimeout(total=self.config.TIMEOUT), allow_redirects=False) as response:
                                if response.status != 429: # 429 Too Many Requests
                                    results.update({"evasion_success": True, "confidence": 0.90})
                                    logger.info(f"Cloudflare rate limit evasion: Succès (status {response.status} != 429) pour {target_url} à la tentative {attempt+1}.")
                                    return results
                                else: # Statut 429 reçu
                                    logger.debug(f"Cloudflare rate limit evasion: Tentative {attempt+1} pour {target_url} a reçu un statut 429.")
                        except asyncio.TimeoutError:
                            logger.debug(f"Cloudflare rate limit evasion: Timeout pour {target_url} (tentative {attempt+1}).")
                            self.packet_loss_count += 1
                        except aiohttp.ClientError as e:
                            logger.debug(f"Cloudflare rate limit evasion: ClientError pour {target_url} (tentative {attempt+1}): {type(e).__name__} - {e}")
                            self.packet_loss_count += 1
                        except Exception as e_gen:
                            logger.error(f"Cloudflare rate limit evasion: Erreur inattendue (tentative {attempt+1}): {e_gen}", exc_info=True)
                            self.packet_loss_count += 1

                    if attempt < self.config.MAX_RETRIES - 1: # Délai avant la prochaine tentative
                        await asyncio.sleep(random.uniform(self.config.RETRY_BACKOFF_FACTOR, self.config.RETRY_BACKOFF_FACTOR * 2))

        except asyncio.TimeoutError: # Timeout global pour toutes les tentatives
            logger.warning(f"Cloudflare rate limit evasion: Timeout global pour {target_url} après {self.config.MAX_RETRIES} tentatives.")
            results.update({"confidence": 0.75, "evasion_success": False, "error": "Global timeout"})
            self.packet_loss_count += 1

        if not results["evasion_success"]:
             logger.info(f"Cloudflare rate limit evasion: Échec de toutes les tentatives pour {target_url}.")
        return results

    async def cloudflare_tcp_fingerprint(self, ip: str, port: int) -> Dict[str, Any]:
        """
        Analyse l'empreinte TCP (spécifiquement le TTL de la réponse) pour inférer la présence de Cloudflare.

        Nécessite des privilèges administratifs pour l'envoi de paquets Scapy.
        Une plage de TTL spécifique (110-120) est souvent associée aux serveurs derrière Cloudflare.

        Args:
            ip (str): L'adresse IP cible.
            port (int): Le port cible.

        Returns:
            Dict[str, Any]: Un dictionnaire avec "inferred_protection" ("cloudflare" ou "none")
                            et "confidence". Retourne "none" si non admin ou en cas d'erreur.
        """
        async with self.semaphore:
            results: Dict[str, Any] = {"inferred_protection": "none", "confidence": 0.0}
        if not self.state.is_admin:
            logger.warning(f"Cloudflare TCP fingerprint: Désactivé pour {ip}:{port}, privilèges administratifs requis.")
            return results # Retourner des résultats neutres si pas admin

        try:
            async with asyncio.timeout(self.config.TIMEOUT): # Timeout pour l'opération Scapy
                # Options TCP courantes pour simuler un client standard
                tcp_options: List[Tuple[str, Any]] = [("MSS", 1460), ("NOP", None), ("WScale", 8), ("NOP", None), ("NOP", None), ("Timestamp", (0,0)), ("SACKPermitted", b'')] # type: ignore
                source_ip: Optional[str] = self._get_random_source_ip()

                # Forger le paquet SYN
                # mypy a du mal avec l'inférence de type complexe de Scapy
                pkt: Any = IP(dst=ip, src=source_ip) / TCP(dport=port, flags="S", options=tcp_options, sport=RandShort()) if source_ip else IP(dst=ip) / TCP(dport=port, flags="S", options=tcp_options, sport=RandShort())

                start_time: float = time.time()
                # sr1 envoie et reçoit un seul paquet. La réponse peut être None.
                response: Optional[IP] = await asyncio.get_event_loop().run_in_executor( # type: ignore
                    self.state.executor, # Utiliser l'executor de l'état si disponible et approprié pour Scapy
                    lambda: sr1(pkt, timeout=self.config.TIMEOUT, verbose=0, iface=self.config.DEFAULT_IFACE if hasattr(self.config, 'DEFAULT_IFACE') else None)
                )
                response_time: float = time.time() - start_time
                self.response_times.append(response_time)
                await asyncio.sleep(random.uniform(0.1, 0.5))  # Délai aléatoire
                if response and response.haslayer(TCP):
                    # Vérifier si le paquet a une couche IP et TCP pour accéder au TTL
                    if response.haslayer(IP):
                        ttl: int = response[IP].ttl
                        # Plage de TTL typique pour Cloudflare (peut varier)
                        if 110 <= ttl <= 120: # Cloudflare TTLs are often in this range.
                            results.update({"inferred_protection": "cloudflare", "confidence": 0.85})
                            logger.debug(f"Cloudflare TCP fingerprint: TTL={ttl} (suggérant Cloudflare) pour {ip}:{port}.")
                        else:
                            results.update({"inferred_protection": "none", "confidence": 0.80}) # TTL ne correspond pas à Cloudflare
                            logger.debug(f"Cloudflare TCP fingerprint: TTL={ttl} (ne suggère pas Cloudflare) pour {ip}:{port}.")
                    else: # Réponse sans couche IP (inattendu pour une réponse TCP)
                        results.update({"inferred_protection": "unknown", "confidence": 0.70, "error": "Réponse sans couche IP."})
                        logger.warning(f"Cloudflare TCP fingerprint: Réponse reçue sans couche IP pour {ip}:{port}.")
                else: # Aucune réponse ou réponse non TCP
                    results.update({"inferred_protection": "unknown", "confidence": 0.75, "error": "Aucune réponse TCP ou réponse non-TCP."})
                    self.packet_loss_count += 1
                    logger.debug(f"Cloudflare TCP fingerprint: Aucune réponse TCP ou réponse non-TCP pour {ip}:{port}.")
        except asyncio.TimeoutError:
            logger.debug(f"Cloudflare TCP fingerprint: Timeout pour {ip}:{port}.")
            results.update({"inferred_protection": "unknown", "confidence": 0.65, "error": "Timeout de l'opération Scapy."})
            self.packet_loss_count +=1
        except Exception as e: # Erreurs Scapy ou autres
            logger.error(f"Erreur lors de la fingerprint TCP pour {ip}:{port}: {e}", exc_info=True)
            results.update({"inferred_protection": "unknown", "confidence": 0.60, "error": f"Erreur Scapy: {e}"})
            self.packet_loss_count += 1
        return results

    async def _network_noise_analysis(self, ip: str, port: int) -> Dict[str, Any]:
        """
        Analyse la gigue (variation des délais de réponse) pour détecter des signes d'IDS/IPS.

        Envoie une série de paquets et mesure l'écart-type des temps de réponse.
        Une forte variation peut indiquer une inspection par un IDS/IPS.
        Nécessite des privilèges administratifs.

        Args:
            ip (str): L'adresse IP cible.
            port (int): Le port cible.

        Returns:
            Dict[str, Any]: Dictionnaire avec "protection" ("ids_ips" ou "none")
                            et "protection_confidence".
        """
        async with self.semaphore:
            results: Dict[str, Any] = {"protection": "none", "protection_confidence": 0.5}
        if not self.state.is_admin:
            logger.warning(f"Analyse de bruit réseau désactivée pour {ip}:{port}, privilèges admin requis.")
            return results # Retour neutre si pas admin

        local_response_times: List[float] = [] # Utiliser une liste locale pour cette analyse spécifique
        num_probes: int = 7 # Nombre de sondes à envoyer

        try:
            async with asyncio.timeout(self.config.TIMEOUT * num_probes / 2): # Timeout global pour l'analyse
                for i in range(num_probes):
                    source_ip: Optional[str] = self._get_random_source_ip()
                    pkt: Any = IP(dst=ip, src=source_ip) / TCP(dport=port, flags="S", sport=RandShort()) if source_ip else IP(dst=ip) / TCP(dport=port, flags="S", sport=RandShort())

                    start_time: float = time.time()
                    # Utiliser un timeout plus court par paquet pour cette analyse
                    response: Optional[IP] = await asyncio.get_event_loop().run_in_executor( # type: ignore
                        self.state.executor,
                        lambda: sr1(pkt, timeout=max(0.1, self.config.TIMEOUT / 2), verbose=0, iface=self.config.DEFAULT_IFACE if hasattr(self.config, 'DEFAULT_IFACE') else None)
                    )
                    response_time: float = time.time() - start_time

                    if response: # Seulement considérer les réponses reçues pour le calcul de la gigue
                        local_response_times.append(response_time)
                    else: # Compter les paquets perdus
                        self.packet_loss_count += 1
                        logger.debug(f"Network noise: Paquet {i+1}/{num_probes} perdu pour {ip}:{port}.")

                    if i < num_probes - 1: # Ne pas attendre après la dernière sonde
                        await asyncio.sleep(random.uniform(0.05, 0.2)) # Court délai entre les sondes

            if len(local_response_times) >= num_probes // 2: # Assez de données pour analyser
                avg_time: float = sum(local_response_times) / len(local_response_times)
                # Calcul de l'écart-type
                variance: float = sum((t - avg_time) ** 2 for t in local_response_times) / len(local_response_times)
                std_dev: float = variance ** 0.5

                # Seuil pour la détection d'IDS/IPS (ex: si l'écart-type est > 15% de la moyenne)
                # Ce seuil est empirique et peut nécessiter des ajustements.
                if avg_time > 0 and std_dev > (0.15 * avg_time):
                    results.update({"protection": "ids_ips_suspected", "protection_confidence": 0.8})
                    logger.info(f"Protection IDS/IPS suspectée sur {ip}:{port} : Écart-type={std_dev:.4f}s, Moyenne={avg_time:.4f}s.")
                else:
                    logger.debug(f"Analyse de bruit réseau pour {ip}:{port} : Écart-type={std_dev:.4f}s, Moyenne={avg_time:.4f}s. Pas d'IDS/IPS détecté par gigue.")
            else:
                logger.warning(f"Analyse de bruit réseau pour {ip}:{port} : Trop peu de réponses ({len(local_response_times)}/{num_probes}) pour une analyse fiable.")
                results["error"] = "Données insuffisantes pour l'analyse de gigue."

        except asyncio.TimeoutError:
            logger.warning(f"Analyse de bruit réseau : Timeout global pour {ip}:{port}.")
            results.update({"protection": "unknown", "protection_confidence": 0.4, "error": "Timeout global de l'analyse."})
            self.packet_loss_count +=1
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse de bruit réseau sur {ip}:{port} : {e}", exc_info=True)
            self.packet_loss_count += 1
            results.update({"protection": "error", "protection_confidence": 0.3, "error": str(e)})
        return results

    async def detect_protection_advanced(self, ip: str, port: int) -> Dict[str, Any]:
        """
        Détecte des protections (pare-feu, IDS/IPS) en envoyant un paquet ACK "anormal".

        Un paquet ACK envoyé à un port qui n'a pas de connexion établie peut provoquer
        des réponses différentes selon la configuration du pare-feu ou de l'IDS/IPS.
        Une réponse RST est normale pour un port fermé. Une absence de réponse ou un ICMP
        peut indiquer un filtrage. Des délais anormaux peuvent suggérer un IDS/IPS.
        Nécessite des privilèges administratifs.

        Args:
            ip (str): L'adresse IP cible.
            port (int): Le port cible.

        Returns:
            Dict[str, Any]: Dictionnaire avec "protection" ("firewall", "ids_ips", "none")
                            et "protection_confidence".
        """
        async with self.semaphore:
            results: Dict[str, Any] = {"protection": "none", "protection_confidence": 0.5}
        if not self.state.is_admin:
            logger.warning(f"Détection de protection avancée désactivée pour {ip}:{port}, privilèges admin requis.")
            return results

        try:
            async with asyncio.timeout(self.config.TIMEOUT * 1.5): # Timeout pour cette opération
                source_ip: Optional[str] = self._get_random_source_ip()
                # Paquet ACK avec un numéro de séquence aléatoire (peu susceptible de correspondre à une connexion existante)
                pkt: Any = IP(dst=ip, src=source_ip) / TCP(dport=port, flags="A", sport=RandShort(), seq=random.randint(1000, 1000000), ack=random.randint(1000,1000000)) if source_ip else IP(dst=ip) / TCP(dport=port, flags="A", sport=RandShort(), seq=random.randint(1000,1000000), ack=random.randint(1000,1000000))

                start_time: float = time.time()
                response: Optional[IP] = await asyncio.get_event_loop().run_in_executor( # type: ignore
                    self.state.executor,
                    lambda: sr1(pkt, timeout=self.config.TIMEOUT, verbose=0, iface=self.config.DEFAULT_IFACE if hasattr(self.config, 'DEFAULT_IFACE') else None)
                )
                response_time: float = time.time() - start_time
            self.response_times.append(response_time)
            await asyncio.sleep(random.uniform(0.1, 0.5))  # Délai aléatoire

            if response:
                if response.haslayer(ICMP) and response[ICMP].type == 3: # ICMP Unreachable
                    results.update({"protection": "firewall_strong", "protection_confidence": 0.85}) # Plus confiant si ICMP explicite
                    logger.debug(f"Protection avancée (pare-feu) détectée sur {ip}:{port} : ICMP type 3 reçu à un paquet ACK.")
                elif response.haslayer(TCP) and (response[TCP].flags == "R" or response[TCP].flags == 0x04): # RST flag
                    # Calculer la moyenne des 5 derniers temps de réponse stockés globalement (si disponibles)
                    # Cela peut être bruité si les scans précédents étaient sur des ports/protocoles différents.
                    # Une liste de temps de réponse locale à cette fonction pourrait être plus précise.
                    if self.response_times and len(self.response_times) >= 5:
                        avg_prev_response_time: float = sum(self.response_times[-5:]) / min(len(self.response_times), 5)
                        # Si le temps de réponse actuel est significativement plus élevé, cela pourrait indiquer une inspection.
                        if response_time > (avg_prev_response_time * 2.5) and avg_prev_response_time > 0.01 : # Seuil empirique
                            results.update({"protection": "ids_ips_suspected_latency", "protection_confidence": 0.75})
                            logger.debug(f"Protection IDS/IPS suspectée sur {ip}:{port} (latence anormale ACK): {response_time:.3f}s vs moyenne {avg_prev_response_time:.3f}s.")
                        else: # Comportement normal (RST rapide)
                            results.update({"protection": "none_or_standard_closed", "protection_confidence": 0.7})
                            logger.debug(f"Protection avancée: Réponse RST normale à un paquet ACK sur {ip}:{port}. Temps: {response_time:.3f}s.")
                    else: # Pas assez de données pour comparer la latence
                        results.update({"protection": "none_or_standard_closed", "protection_confidence": 0.65})
                        logger.debug(f"Protection avancée: Réponse RST normale à un paquet ACK sur {ip}:{port} (pas de données de latence comparative).")
                # Si aucune réponse (filtré par un pare-feu stateful qui ignore les ACK non sollicités)
                elif not response:
                    results.update({"protection": "firewall_stateful_or_filtered", "protection_confidence": 0.8})
                    self.packet_loss_count += 1
                    logger.debug(f"Protection avancée (pare-feu stateful?) détectée sur {ip}:{port} : Aucune réponse à un paquet ACK.")
                else: # Réponse inattendue
                    results.update({"protection": "unknown_response_to_ack", "protection_confidence": 0.6})
                    logger.debug(f"Protection avancée: Réponse inattendue ({response.summary()}) à un paquet ACK sur {ip}:{port}.")

            else: # Aucune réponse du tout (timeout de sr1)
                 results.update({"protection": "filtered_no_response_ack", "protection_confidence": 0.7})
                 self.packet_loss_count += 1
                 logger.debug(f"Protection avancée: Aucune réponse (timeout sr1) à un paquet ACK sur {ip}:{port}.")

        except asyncio.TimeoutError:
            logger.warning(f"Détection de protection avancée : Timeout global pour {ip}:{port}.")
            results.update({"protection": "unknown_timeout", "protection_confidence": 0.4, "error": "Timeout global de l'analyse."})
            self.packet_loss_count +=1
        except Exception as e:
            logger.error(f"Erreur lors de la détection de protection avancée sur {ip}:{port} : {e}", exc_info=True)
            results.update({"protection": "error", "protection_confidence": 0.3, "error": str(e)})
            self.packet_loss_count += 1
        return results

    async def do_full_scan_on_port(self, ip: str, port: int, domain: Optional[str] = None, zombie_ip: Optional[str] = None) -> Dict[str, Any]:
        """
        Orchestre un scan complet et approfondi sur une seule combinaison IP/port.

        Combine plusieurs techniques de scan (TCP Connect, SYN), récupération de bannière,
        identification de version, analyse SSL, énumération de chemins web,
        détection de WAF/Cloudflare, et vérification de vulnérabilités.

        Args:
            ip (str): L'adresse IP cible.
            port (int): Le port cible.
            domain (Optional[str]): Le nom de domaine associé (important pour SNI, Host header, etc.).
            zombie_ip (Optional[str]): Adresse IP d'un hôte zombie pour un scan Idle (non implémenté).

        Returns:
            Dict[str, Any]: Un dictionnaire agrégé contenant tous les résultats
                            des différentes étapes de scan pour ce port.
        """
        async with self.semaphore: # Respecter la limite de concurrence globale pour chaque appel à do_full_scan_on_port
            # Initialisation du dictionnaire de résultats pour ce port spécifique
            port_results: Dict[str, Any] = {
                "ip": ip, "port": port, "status": "error", "service": "unknown",
                "inferred_service": "unknown", "inferred_state": "unknown", "banner": "",
                "version": "", "confidence": 0.1, "protection": "unknown",
                "protection_confidence": 0.0, "bypass_method": "none",
                "vulnerabilities": [], "sensitive_paths": [], "ssl_details": {},
                "subdomains": [], "waf_bypass_info": {}, "cloudflare_info": {},
                "os_guess": "unknown", "errors": [] # Liste pour collecter les erreurs spécifiques à ce port
            }
        try:
            logger.info(f"Démarrage du scan complet pour {ip}:{port} (Domaine: {domain})")

            # Étape 0: Vérification de la résolution DNS si un domaine est fourni.
            # Cela permet de s'assurer que l'IP scannée correspond bien au domaine.
            if domain and not re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", domain): # Si `domain` est un nom d'hôte et non une IP
                try:
                    resolved_ips_set: Set[str] = await self.resolve_domain(domain)
                    if resolved_ips_set and ip not in resolved_ips_set:
                        # Avertissement si l'IP fournie ne correspond pas aux IPs résolues pour le domaine.
                        warning_msg = f"L'IP {ip} ne correspond pas aux IPs résolues pour {domain}: {resolved_ips_set}. Scan continue sur {ip}."
                        logger.warning(warning_msg)
                        port_results["errors"].append(warning_msg)
                except Exception as e_resolve:
                    error_msg = f"Erreur lors de la vérification de la résolution DNS pour {domain}: {e_resolve}"
                    logger.error(error_msg)
                    port_results["errors"].append(error_msg)


            # Étape 1: Scan TCP Connect de base pour déterminer l'état initial du port.
            tcp_scan_results: Dict[str, Any] = await self.tcp_connect(ip, port)
            port_results.update({ # Mettre à jour les résultats avec les infos du scan TCP
                "status": tcp_scan_results.get("status", "error"),
                "service": tcp_scan_results.get("service", "unknown"),
                "confidence": tcp_scan_results.get("confidence", 0.5),
                "protection": tcp_scan_results.get("protection", "none"),
                "protection_confidence": tcp_scan_results.get("protection_confidence", 0.5),
                "os_guess": tcp_scan_results.get("os_guess", "unknown")
            })
            logger.info(f"Résultat TCP connect pour {ip}:{port} : Statut={port_results['status']}, Service={port_results['service']}")

            # Étape 2: Scan SYN si les privilèges admin sont disponibles.
            # Peut affiner le statut du port (ex: distinguer 'closed' de 'filtered') et l'OS.
            if self.state.is_admin:
                syn_scan_results: Dict[str, Any] = await self.syn_scan(ip, port)
                logger.info(f"Résultat SYN scan pour {ip}:{port} : Statut={syn_scan_results.get('status')}, Service={syn_scan_results.get('service')}")
                # Mettre à jour les résultats si le scan SYN fournit une meilleure information (port ouvert ou plus grande confiance)
                if (syn_scan_results.get("status") == "open" and port_results["status"] != "open") or \
                   (syn_scan_results.get("status") == "open" and syn_scan_results.get("confidence", 0.0) > port_results.get("confidence", 0.0)):
                    port_results.update({
                        "status": syn_scan_results["status"], # Le statut du scan SYN prévaut s'il est 'open'
                        "service": syn_scan_results.get("service", port_results["service"]),
                        "confidence": syn_scan_results.get("confidence", port_results["confidence"]),
                        "bypass_method": syn_scan_results.get("bypass_method", port_results["bypass_method"]),
                        "protection": syn_scan_results.get("protection", port_results["protection"]),
                        "os_guess": syn_scan_results.get("os_guess", port_results["os_guess"])
                    })
                    logger.info(f"Scan SYN a mis à jour le statut/service pour {ip}:{port}.")
                elif port_results["status"] != "open" and syn_scan_results.get("status") == "filtered":
                    # Si TCP Connect a trouvé 'closed' et SYN trouve 'filtered', 'filtered' est plus précis.
                    port_results["status"] = "filtered"


            # Étape 3: Actions supplémentaires si le port est détecté comme ouvert.
            if port_results["status"] == "open":
                # Récupération de la bannière du service
                banner_grab_results: Dict[str, Any] = await self.grab_banner(ip, port, domain)
                logger.info(f"Bannière pour {ip}:{port} : {banner_grab_results.get('banner', '')[:60]}...") # Log tronqué
                port_results.update({
                    "banner": banner_grab_results.get("banner", ""),
                    "service": banner_grab_results.get("service", port_results["service"]) if banner_grab_results.get("service") != "unknown" else port_results["service"], # Le service de la bannière peut être plus précis
                    "inferred_service": banner_grab_results.get("inferred_service", port_results["inferred_service"]),
                    "confidence": max(port_results.get("confidence", 0.0), banner_grab_results.get("confidence", 0.0))
                })

                # Identification de la version du service à partir de la bannière
                if port_results["banner"]:
                    service_version_results: Dict[str, Any] = await self.identify_service_version(ip, port, port_results["banner"])
                    port_results["version"] = service_version_results.get("version", "")
                    port_results["confidence"] = max(port_results.get("confidence", 0.0), service_version_results.get("confidence", 0.0))

                # Vérification des vulnérabilités connues si une version ou une bannière est disponible
                if port_results["version"] or port_results["banner"]:
                    vulnerabilities_found: List[Dict[str, Any]] = await self.check_vulnerabilities(ip, port, port_results["version"], port_results["banner"])
                    port_results["vulnerabilities"] = vulnerabilities_found

            # Étape 4: Analyses spécifiques aux ports web (HTTP/HTTPS).
            service_lower: str = str(port_results.get("service", "")).lower()
            # Conditions pour déterminer si le service est HTTP ou HTTPS
            is_http_service: bool = "http" in service_lower or (port_results["status"] == "open" and port in [80, 8080])
            is_https_service: bool = "https" in service_lower or "ssl" in service_lower or (port_results["status"] == "open" and port in [443, 8443])

            if is_http_service or is_https_service:
                # Tentatives de contournement de WAF
                waf_bypass_results: Dict[str, Any] = await self.waf_bypass(ip, port, domain)
                port_results["waf_bypass_info"] = waf_bypass_results
                logger.info(f"Résultat WAF bypass pour {ip}:{port} : Succès={waf_bypass_results.get('bypass_success')}, WAF={waf_bypass_results.get('waf_detected')}")
                if waf_bypass_results.get("bypass_success"): # Si le bypass a réussi, mettre à jour les infos de protection
                    port_results["protection"] = waf_bypass_results.get("waf_detected", port_results["protection"])
                    port_results["protection_confidence"] = max(port_results.get("protection_confidence", 0.0), waf_bypass_results.get("confidence", 0.0))

                # Sondes spécifiques à Cloudflare
                cloudflare_probe_results: Dict[str, Any] = await self.cloudflare_stealth_probe(ip, port, domain)
                port_results["cloudflare_info"] = cloudflare_probe_results
                logger.info(f"Résultat Cloudflare probe pour {ip}:{port} : Bypass={cloudflare_probe_results.get('bypass_success')}, Protection={cloudflare_probe_results.get('protection')}")

                if cloudflare_probe_results.get("bypass_success"): # Si la sonde Cloudflare a réussi
                    if cloudflare_probe_results.get("protection") != "none": # Mettre à jour la protection si Cloudflare détecté
                        port_results["protection"] = cloudflare_probe_results.get("protection", port_results["protection"])
                        port_results["protection_confidence"] = max(port_results.get("protection_confidence", 0.0), cloudflare_probe_results.get("confidence", 0.0))
                else: # Si la sonde Cloudflare a été bloquée
                    # Tenter de résoudre un défi JavaScript si Cloudflare est suspecté ou si le port est ouvert
                    if "cloudflare" in str(cloudflare_probe_results.get("protection","")).lower() or port_results.get("status") == "open":
                        logger.info(f"Tentative de résolution de défi JS pour {ip}:{port} car la sonde Cloudflare a été bloquée ou le port est ouvert.")
                        js_challenge_results: Dict[str, Any] = await self.cloudflare_js_challenge_solver(ip, port, domain)
                        port_results["cloudflare_info"]["js_challenge_solver"] = js_challenge_results # Nicher les résultats du solveur
                        if js_challenge_results.get("bypass_success"): # Si le défi JS est résolu
                             port_results["protection"] = "cloudflare_js_bypassed"
                             port_results["protection_confidence"] = js_challenge_results.get("confidence", 0.9)

                # Énumération des chemins web si le port est ouvert
                if port_results["status"] == "open":
                    web_paths_results: List[Dict[str, Any]] = await self.enumerate_web_paths(ip, port, domain)
                    port_results["sensitive_paths"] = web_paths_results

                # Analyse SSL si le service est HTTPS et le port est ouvert
                if is_https_service and port_results["status"] == "open":
                    ssl_analysis_results: Dict[str, Any] = await self.advanced_ssl_analysis(ip, port, domain)
                    port_results["ssl_details"] = ssl_analysis_results
                    logger.info(f"Résultat SSL analysis pour {ip}:{port} : Émetteur={ssl_analysis_results.get('issuer')}, SANs={ssl_analysis_results.get('san_domains')}")

            # Étape 5: Énumération des sous-domaines si un nom de domaine est fourni.
            # Cette étape est coûteuse et peut être conditionnée pour éviter la redondance.
            if domain and not re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", domain):
                # Conditionner l'énumération, par ex., seulement pour les ports web principaux (80, 443) ou si un service web est détecté.
                if port in [80, 443] or "http" in port_results.get("service","").lower():
                    subdomain_enum_results: List[str] = await self.advanced_subdomain_enumeration(ip, domain)
                    port_results["subdomains"] = subdomain_enum_results
                    logger.info(f"Sous-domaines trouvés pour {domain} (associés à {ip}): {subdomain_enum_results}")

            # Étape 6: Scan Zombie (Idle Scan) - Actuellement non implémenté.
            if zombie_ip:
                logger.warning(f"Scan zombie (Idle Scan) avec zombie {zombie_ip} non implémenté pour {ip}:{port}.")
                port_results["errors"].append(f"Scan zombie non implémenté pour zombie {zombie_ip}")


            # Mise à jour finale de l'état inféré du port pour un affichage clair.
            if port_results["status"] == "open":
                port_results["inferred_state"] = "ouvert"
            elif port_results["status"] == "closed":
                port_results["inferred_state"] = "fermé"
            elif port_results["status"] == "filtered":
                 port_results["inferred_state"] = "filtré"
            else: # Cas d'erreur ou autre statut
                 port_results["inferred_state"] = f"état_incertain ({port_results['status']})"


            # Log si des pertes de paquets ont été enregistrées pendant le scan de ce port.
            # Note: `packet_loss_count` est un compteur global. Pour un suivi par port, il faudrait une logique différente.
            if self.packet_loss_count > port_results.get("_initial_packet_loss_count", 0):
                logger.warning(f"Des pertes de paquets ou des erreurs réseau ont été détectées pendant le scan de {ip}:{port}.")
                # On pourrait ajouter un marqueur spécifique dans les résultats du port si nécessaire.
                # port_results["network_issues_detected"] = True

        except Exception as e_main_scan: # Gérer les erreurs majeures pendant l'orchestration du scan du port.
            error_message = f"Erreur majeure et inattendue lors du scan complet de {ip}:{port} : {type(e_main_scan).__name__} - {e_main_scan}"
            logger.error(error_message, exc_info=True)
            port_results["errors"].append(error_message) # Ajouter l'erreur aux résultats du port
            port_results["status"] = "error_orchestration" # Statut spécifique pour indiquer une erreur d'orchestration

        logger.info(f"Scan complet terminé pour {ip}:{port}. Statut final: {port_results['status']}, Service: {port_results['service']}")
        return port_results

    async def scan_all_ports(self, ip: str, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Scanne tous les ports TCP de 1 à 65535 sur une adresse IP donnée.

        Orchestre l'appel de `do_full_scan_on_port` pour chaque port, en parallélisant
        les opérations jusqu'à `APP_CONFIG.MAX_PARALLEL`.

        Args:
            ip (str): L'adresse IP à scanner.
            domain (Optional[str]): Le nom de domaine associé (passé à `do_full_scan_on_port`).

        Returns:
            List[Dict[str, Any]]: Une liste de dictionnaires, où chaque dictionnaire
                                  contient les résultats du scan complet pour un port.
                                  Les erreurs de scan de port individuel sont loggées
                                  mais peuvent ne pas être incluses ici si elles lèvent des exceptions.
        """
        # Créer une liste de tâches pour tous les ports
        # Le range va de 1 à 65535 inclusivement.
        # Note: Scanner tous les ports est extrêmement long et peut être détecté.
        # Pour des tests, utiliser une plage de ports plus restreinte.
        # common_ports = [21, 22, 23, 25, 53, 80, 110, 143, 443, 445, 3306, 3389, 5432, 5900, 8080, 8443]
        # tasks = [self.do_full_scan_on_port(ip, port, domain) for port in common_ports]
        tasks: List[asyncio.Task] = [
            asyncio.create_task(self.do_full_scan_on_port(ip, port, domain))
            for port in range(1, 65536) # Scan de 1 à 65535
        ]

        all_results: List[Dict[str, Any]] = []

        # Traiter les tâches par lots pour gérer la concurrence et la mémoire
        for i in range(0, len(tasks), self.config.MAX_PARALLEL):
            current_batch_tasks: List[asyncio.Task] = tasks[i:i + self.config.MAX_PARALLEL]
            # `asyncio.gather` exécute les tâches du lot en parallèle.
            # `return_exceptions=True` permet de récupérer les exceptions au lieu de les propager immédiatement.
            batch_scan_results: List[Union[Dict[str, Any], Exception]] = await asyncio.gather(*current_batch_tasks, return_exceptions=True)

            for res_or_exc in batch_scan_results:
                if isinstance(res_or_exc, dict):
                    # Filtrer les ports non ouverts ou non pertinents si nécessaire avant d'ajouter
                    # if res_or_exc.get("status") == "open" or res_or_exc.get("vulnerabilities"):
                    all_results.append(res_or_exc)
                elif isinstance(res_or_exc, Exception):
                    # Logguer l'erreur pour un port spécifique.
                    # L'erreur a déjà dû être loggée dans do_full_scan_on_port,
                    # mais un log de haut niveau ici peut être utile.
                    logger.error(f"Erreur lors du scan d'un port dans un lot pour {ip} (domaine: {domain}): {type(res_or_exc).__name__} - {res_or_exc}", exc_info=False) # exc_info=False pour concision
                    # On pourrait ajouter un placeholder ou un enregistrement d'erreur à all_results si nécessaire.
                    # Par exemple: all_results.append({"ip": ip, "port": "unknown_due_to_error", "error_details": str(res_or_exc)})

            # Délai optionnel entre les lots pour réduire la charge ou la détection.
            if i + self.config.MAX_PARALLEL < len(tasks): # S'il y a d'autres lots à venir
                delay_between_batches = random.uniform(0.2, 0.8) # Ex: 0.2s à 0.8s
                logger.debug(f"Scan de lot terminé pour {ip}. Attente de {delay_between_batches:.2f}s avant le prochain lot.")
                await asyncio.sleep(delay_between_batches)

        logger.info(f"Scan de tous les ports terminé pour {ip}. {len(all_results)} résultats collectés.")
        return all_results

# Exemple d'utilisation
async def main():
    state = ScannerState(is_admin=True, executor=None) # type: ignore # Pour l'exemple, executor peut être None
    # Pour une utilisation réelle, initialiser un ThreadPoolExecutor:
    # from concurrent.futures import ThreadPoolExecutor
    # executor = ThreadPoolExecutor(max_workers=APP_CONFIG.MAX_PARALLEL)
    # state = ScannerState(is_admin=True, executor=executor)

    scanner = NetworkScanner(state)
    logger.info("Test recommandé : scan sur scanme.nmap.org ou IP directe comme 8.8.8.8")

    target_ip_for_test = "scanme.nmap.org" # Utiliser un domaine pour tester SNI etc.
    target_port_for_test = 80
    # Alternative: target_ip_for_test = "8.8.8.8", target_port_for_test = 53

    # Pour tester le scan de tous les ports (long!)
    # results_all_ports = await scanner.scan_all_ports(target_ip_for_test, domain=target_ip_for_test if not re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", target_ip_for_test) else None)
    # print(f"\n--- Résultats du scan de tous les ports pour {target_ip_for_test} ---")
    # for res_port in results_all_ports:
    #     if res_port.get("status") == "open": # Afficher seulement les ports ouverts pour la concision
    #         print(f"Port {res_port['port']}: {res_port['status']}, Service: {res_port.get('service', 'N/A')}, Version: {res_port.get('version', 'N/A')}")

    # Scan d'un seul port pour un test plus rapide
    result_single_port = await scanner.do_full_scan_on_port(target_ip_for_test, target_port_for_test, domain=target_ip_for_test if not re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", target_ip_for_test) else None)
    print(f"\n--- Résultat du scan complet pour {target_ip_for_test}:{target_port_for_test} ---")
    import json
    print(json.dumps(result_single_port, indent=4, ensure_ascii=False))

    # N'oubliez pas de fermer l'executor si vous en avez créé un
    # if hasattr(state, 'executor') and state.executor is not None:
    #     state.executor.shutdown(wait=True)

if __name__ == "__main__":
    # Configuration du logging pour l'exemple
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    logger.setLevel(logging.DEBUG) # Mettre DEBUG pour voir plus de détails du scanner

    # Note: Pour exécuter du code asyncio avec des privilèges (comme pour Scapy),
    # le script lui-même doit être lancé avec sudo.
    # asyncio.run() est le point d'entrée standard.
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Scan interrompu par l'utilisateur.")
    except Exception as e_global:
        logger.critical(f"Erreur globale non gérée dans main: {e_global}", exc_info=True)
    results = []
    for i in range(0, len(tasks), self.config.MAX_PARALLEL):
        batch = tasks[i:i + self.config.MAX_PARALLEL]
        batch_results = await asyncio.gather(*batch, return_exceptions=True)
        for res in batch_results:
            if isinstance(res, dict):
                results.append(res)
            else:
                logger.error(f"Erreur dans le scan d'un port : {res}")
        await asyncio.sleep(random.uniform(0.1, 0.5))  # Délai aléatoire entre batches
    return results

# Exemple d'utilisation
async def main():
    state = ScannerState(is_admin=True, executor=None)
    scanner = NetworkScanner(state)
    # Test avec une cible sans WAF pour validation (recommandation)
    logger.info("Test recommandé : scan sur scanme.nmap.org ou IP directe comme 8.8.8.8")
    result = await scanner.do_full_scan_on_port("scanme.nmap.org", 80, domain="scanme.nmap.org")  # Test avec IP nmap directe
    #result = await scanner.do_full_scan_on_port("8.8.8.8", 53, domain=None)  # Test avec IP directe
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
