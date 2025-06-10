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
    """Classe pour effectuer des scans réseau avancés avec rotation d'IP."""
    
    def __init__(self, state: ScannerState):
        if not hasattr(state, 'is_admin'):
            raise ValueError("L'objet state doit contenir l'attribut 'is_admin'")
        if not hasattr(state, 'executor'):
            raise ValueError("L'objet state doit contenir l'attribut 'executor'")
        self.state = state
        self.semaphore = asyncio.Semaphore(APP_CONFIG.MAX_PARALLEL)
        self.config = APP_CONFIG
        self.lstm_model = self._load_lstm_model()
        self.service_signatures = {
            "ssh": {"window": [0, 65535], "ttl": [64, 128], "flags": ["SA", "R"], "confidence": 0.7},
            "http": {"window": [8192, 32768], "ttl": [64, 128], "flags": ["SA", "R"], "confidence": 0.6},
            "firewall": {"icmp_type": 3, "confidence": 0.8}
        }
        self.response_times = []
        self.proxies = self._load_proxies()
        self.current_proxy_index = 0
        self.source_ips = self.config.SOURCE_IPS if hasattr(self.config, 'SOURCE_IPS') else []
        self.packet_loss_count = 0  # Ajout pour surveiller les pertes de paquets

    def _load_lstm_model(self) -> Optional[LSTMModel]:
        """Charge le modèle LSTM à partir de data.pkl ou initialise un nouveau modèle si le fichier est introuvable."""
        try:
            model = LSTMModel(input_size=6, hidden_size=128, num_layers=2, output_size=10)
            model_path = "data.pkl"
            if torch.cuda.is_available():
                model = model.cuda()
                state_dict = torch.load(model_path, map_location=torch.device('cuda'))
            else:
                state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            model.eval()
            logger.info(f"Modèle LSTM chargé avec succès depuis {model_path}")
            return model
        except FileNotFoundError:
            logger.error(f"Échec du chargement des poids du modèle LSTM : Fichier {model_path} introuvable. Initialisation avec des poids aléatoires.")
            model = LSTMModel(input_size=6, hidden_size=128, num_layers=2, output_size=10)
            if torch.cuda.is_available():
                model = model.cuda()
            model.eval()
            return model
        except Exception as e:
            logger.error(f"Erreur inattendue lors du chargement du modèle LSTM : {e}", exc_info=True)
            return None

    def _load_proxies(self) -> List[Dict[str, str]]:
        """Charge une liste de proxies depuis la configuration ou un fichier."""
        try:
            proxies = getattr(self.config, 'PROXIES', [
                {"http": "http://proxy1.example.com:8080", "https": "https://proxy1.example.com:8080"},
                {"http": "http://proxy2.example.com:8080", "https": "https://proxy2.example.com:8080"},
            ])
            if not proxies:
                logger.warning("Aucun proxy configuré, rotation d'IP désactivée pour les requêtes HTTP/HTTPS")
            return proxies
        except Exception as e:
            logger.error(f"Erreur lors du chargement des proxies : {e}", exc_info=True)
            return []

    def _get_next_proxy(self) -> Optional[Dict[str, str]]:
        """Sélectionne le prochain proxy dans la liste."""
        if not self.proxies:
            return None
        proxy = self.proxies[self.current_proxy_index % len(self.proxies)]
        self.current_proxy_index += 1
        logger.debug(f"Utilisation du proxy : {proxy}")
        return proxy

    def _get_random_source_ip(self) -> Optional[str]:
        """Sélectionne une adresse IP source aléatoire pour Scapy."""
        if not self.source_ips:
            return None
        source_ip = random.choice(self.source_ips)
        logger.debug(f"Utilisation de l'IP source : {source_ip}")
        return source_ip

    def _extract_features(self, pkt, response_time: float) -> torch.Tensor:
        """Extrait les caractéristiques réseau pour la prédiction."""
        try:
            features = [
                pkt[IP].ttl if pkt and pkt.haslayer(IP) else 64.0,
                pkt[TCP].window if pkt and pkt.haslayer(TCP) else 0.0,
                pkt[IP].len if pkt and pkt.haslayer(IP) else 0.0,
                response_time,
                1.0 if pkt and pkt.haslayer(ICMP) else 0.0,
                len(pkt[TCP].options) if pkt and pkt.haslayer(TCP) else 0.0
            ]
            return torch.tensor([features], dtype=torch.float32)
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction des caractéristiques : {e}", exc_info=True)
            return torch.tensor([[64.0, 0.0, 0.0, response_time, 0.0, 0.0]], dtype=torch.float32)

    def _predict_service_lstm(self, features: torch.Tensor) -> tuple[str, float]:
        """Prédit le service avec le modèle LSTM."""
        if not self.lstm_model:
            return "unknown", 0.5
        try:
            with torch.no_grad():
                output = self.lstm_model(features)
                service_idx = torch.argmax(output, dim=1).item()
                confidence = torch.softmax(output, dim=1)[0][service_idx].item()
                services = ["http", "ssh", "ftp", "smtp", "telnet", "dns", "mysql", "rdp", "sip", "unknown"]
                return services[min(service_idx, len(services)-1)], confidence
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction LSTM : {e}", exc_info=True)
            return "unknown", 0.5

    def _predict_service_heuristic(self, pkt, response_time: float) -> tuple[str, float]:
        """Prédit le service via des règles heuristiques."""
        try:
            if pkt and pkt.haslayer(ICMP) and pkt[ICMP].type == 3:
                return "filtered", self.service_signatures["firewall"]["confidence"]
            if pkt and pkt.haslayer(TCP):
                window = pkt[TCP].window
                ttl = pkt[IP].ttl if pkt.haslayer(IP) else 64
                flags = pkt[TCP].flags
                for service, sig in self.service_signatures.items():
                    if service == "firewall":
                        continue
                    if (sig["window"][0] <= window <= sig["window"][1] and
                        sig["ttl"][0] <= ttl <= sig["ttl"][1] and
                        str(flags) in sig["flags"]):
                        return service, sig["confidence"]
            return "unknown", 0.5
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction heuristique : {e}", exc_info=True)
            return "unknown", 0.5

    async def resolve_domain(self, domain: str) -> Set[str]:
        """Résout un domaine en une liste d'adresses IP."""
        ip_set = set()
        try:
            answers = await asyncio.get_event_loop().run_in_executor(None, lambda: dns.resolver.resolve(domain, 'A'))
            for rdata in answers:
                ip_set.add(rdata.address)
            logger.debug(f"Résolution DNS réussie pour {domain} : {ip_set}")
            # Ajout de vérification explicite des IPs résolues
            if not ip_set:
                logger.warning(f"Aucune adresse IP résolue pour {domain}. Vérifiez le domaine ou utilisez une IP directe.")
            else:
                logger.info(f"Adresses IP résolues pour {domain} : {', '.join(ip_set)}")
        except dns.resolver.NXDOMAIN:
            logger.warning(f"Domaine {domain} non trouvé.")
        except dns.resolver.NoAnswer:
            logger.warning(f"Aucune réponse DNS pour {domain}.")
        except Exception as e:
            logger.error(f"Erreur lors de la résolution DNS de {domain} : {e}", exc_info=True)
        return ip_set

    async def tcp_connect(self, ip: str, port: int) -> Dict:
        """Effectue un scan TCP connect."""
        async with self.semaphore:
            results = {"status": "closed", "service": "unknown", "confidence": 0.5, "protection": "none", "os_guess": "unknown"}
            for attempt in range(self.config.MAX_RETRIES):
                try:
                    async with asyncio.timeout(self.config.TIMEOUT):
                        reader, writer = await asyncio.open_connection(ip, port)
                        results.update({"status": "open", "confidence": 0.9, "protection": "none"})
                        writer.close()
                        await writer.wait_closed()
                        logger.info(f"Scan TCP connect : {ip}:{port} ouvert (tentative {attempt+1})")
                        return results
                except (asyncio.TimeoutError, ConnectionRefusedError, OSError) as e:
                    logger.debug(f"Scan TCP connect : {ip}:{port} fermé ou filtré (tentative {attempt+1}) : {e}")
                    self.packet_loss_count += 1  # Surveillance des pertes
                    if attempt < self.config.MAX_RETRIES - 1:
                        await asyncio.sleep(random.uniform(0.1, 0.5))  # Délai aléatoire
            closed_results = await self.probe_closed_port_advanced(ip, port)
            results.update(closed_results)
            if results["status"] == "error":
                results["status"] = "closed"
                logger.warning(f"Statut 'error' corrigé en 'closed' pour {ip}:{port}")
            return results

async def probe_closed_port_advanced(self, ip: str, port: int) -> Dict:
    """Sonde avancée pour détecter services et protections sur ports fermés."""
    async with self.semaphore:
        results = {"status": "closed", "service": "unknown", "confidence": 0.5, "protection": "none", "os_guess": "unknown"}
        for attempt in range(self.config.MAX_RETRIES):
            try:
                source_ip = self._get_random_source_ip()
                pkt = IP(dst=ip, src=source_ip) / TCP(dport=port, flags="S", sport=RandShort()) if source_ip else IP(dst=ip) / TCP(dport=port, flags="S", sport=RandShort())
                start_time = time.time()
                response = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: sr1(pkt, timeout=self.config.TIMEOUT, verbose=0)
                )
                response_time = time.time() - start_time
                self.response_times.append(response_time)
                logger.debug(f"Sonde SYN sur {ip}:{port} : réponse en {response_time:.3f}s")

                if response:
                    if response.haslayer(TCP) and response[TCP].flags == "SA":  # SYN-ACK
                        results.update({"status": "open", "confidence": 0.95})
                        service, confidence = self._predict_service_heuristic(response, response_time)
                        if self.lstm_model:
                            features = self._extract_features(response, response_time)
                            lstm_service, lstm_confidence = self._predict_service_lstm(features)
                            if lstm_confidence > confidence:
                                service, confidence = lstm_service, lstm_confidence
                        results.update({"service": service, "confidence": confidence})
                        os_guess = await self.fingerprint_tcp_ip(response)
                        results["os_guess"] = os_guess
                        logger.info(f"Port {ip}:{port} détecté comme ouvert avec service {service}")
                        return results
                    elif response.haslayer(TCP) and response[TCP].flags == "R":  # RST
                        results.update({"status": "closed", "confidence": 0.9})
                        logger.debug(f"Port {ip}:{port} détecté comme fermé (RST)")
                    elif response.haslayer(ICMP) and response[ICMP].type == 3:  # ICMP Port Unreachable
                        results.update({"status": "filtered", "service": "filtered", "confidence": 0.7, "protection": "firewall"})
                        logger.debug(f"Port {ip}:{port} détecté comme filtré (ICMP type 3)")
                    else:
                        results.update({"status": "filtered", "confidence": 0.7})
                        logger.debug(f"Port {ip}:{port} détecté comme filtré (réponse inattendue)")
                else:
                    results.update({"status": "filtered", "confidence": 0.7})
                    self.packet_loss_count += 1  # Surveillance des pertes
                    logger.debug(f"Port {ip}:{port} : aucune réponse, possible perte de paquet")

                # Sonde UDP pour des ports comme 53 (DNS)
                udp_results = await self._send_udp_probe(ip, port)
                if udp_results["service"] != "unknown":
                    results.update(udp_results)
                    logger.info(f"Port {ip}:{port} : service UDP détecté - {udp_results['service']}")
                    return results

                return results
            except Exception as e:
                logger.error(f"Erreur lors de la sonde avancée sur {ip}:{port} (tentative {attempt+1}) : {e}", exc_info=True)
                self.packet_loss_count += 1  # Surveillance des pertes
                if attempt < self.config.MAX_RETRIES - 1:
                    await asyncio.sleep(random.uniform(0.1, 0.5))  # Délai aléatoire
        logger.warning(f"Scan avancé échoué pour {ip}:{port} après {self.config.MAX_RETRIES} tentatives")
        return results

async def _send_udp_probe(self, ip: str, port: int) -> Dict:
    """Envoie des sondes UDP pour détecter des services."""
    async with self.semaphore:
        results = {"service": "unknown", "confidence": 0.5}
        try:
            payloads = {
                "dns": b"\x00\x01\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00\x07example\x03com\x00\x00\x01\x00\x01",
                "snmp": b"\x30\x26\x02\x01\x00\x04\x06\x70\x75\x62\x6c\x69\x63\xa0\x19\x02\x04\x00\x00\x00\x01\x02\x01\x00\x02\x01\x00\x30\x0b\x30\x09\x06\x05\x2b\x06\x01\x02\x01\x01\x00",
                "sip": b"OPTIONS sip:user@shopify.com SIP/2.0\r\nVia: SIP/2.0/UDP 127.0.0.1\r\n\r\n"
            }
            for proto, payload in payloads.items():
                source_ip = self._get_random_source_ip()
                pkt = IP(dst=ip, src=source_ip) / UDP(dport=port, sport=RandShort()) / payload if source_ip else IP(dst=ip) / UDP(dport=port, sport=RandShort()) / payload
                start_time = time.time()
                response = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: sr1(pkt, timeout=self.config.TIMEOUT, verbose=0)
                )
                response_time = time.time() - start_time
                self.response_times.append(response_time)
                await asyncio.sleep(random.uniform(0.1, 0.5))  # Délai aléatoire
                if response and response.haslayer(UDP):
                    results.update({"service": proto, "confidence": 0.9})
                    logger.debug(f"Sonde UDP {proto} sur {ip}:{port} : réponse UDP reçue")
                    break
                elif response and response.haslayer(ICMP) and response[ICMP].type == 3:
                    logger.debug(f"Sonde UDP {proto} sur {ip}:{port} : ICMP Port Unreachable")
                else:
                    self.packet_loss_count += 1  # Surveillance des pertes
                    logger.debug(f"Sonde UDP {proto} sur {ip}:{port} : aucune réponse")
        except Exception as e:
            logger.error(f"Erreur lors de la sonde UDP sur {ip}:{port} : {e}", exc_info=True)
            self.packet_loss_count += 1
        return results

async def fingerprint_tcp_ip(self, pkt) -> str:
    """Crée une empreinte TCP/IP pour deviner l'OS."""
    try:
        if not pkt or not pkt.haslayer(IP):
            return "unknown"
        ttl = pkt[IP].ttl
        os_signatures = {
            "linux": {"ttl": [60, 64]},
            "windows": {"ttl": [120, 128]},
            "cisco": {"ttl": [250, 255]},
            "cloudflare": {"ttl": [110, 120]}
        }
        for os, sig in os_signatures.items():
            if sig["ttl"][0] <= ttl <= sig["ttl"][1]:
                logger.debug(f"OS détecté : {os} (TTL={ttl})")
                return os
        return "unknown"
    except Exception as e:
        logger.error(f"Erreur lors de l'empreinte TCP/IP : {e}", exc_info=True)
        return "unknown"

async def syn_scan(self, ip: str, port: int) -> Dict:
    """Effectue un scan SYN (nécessite des privilèges admin)."""
    async with self.semaphore:
        results = {"status": "closed", "service": "unknown", "confidence": 0.5, "bypass_method": "none", "protection": "none", "os_guess": "unknown"}
        if not self.state.is_admin:
            logger.warning(f"Scan SYN ignoré pour {ip}:{port} : privilèges administratifs requis. Passage à TCP connect.")
            return await self.tcp_connect(ip, port)  # Fallback vers TCP connect
        try:
            source_ip = self._get_random_source_ip()
            packet = IP(dst=ip, src=source_ip) / TCP(dport=port, flags="S", sport=RandShort()) if source_ip else IP(dst=ip) / TCP(dport=port, flags="S", sport=RandShort())
            start_time = time.time()
            response = await asyncio.get_event_loop().run_in_executor(None, lambda: sr1(packet, timeout=self.config.TIMEOUT, verbose=0))
            response_time = time.time() - start_time
            self.response_times.append(response_time)
            await asyncio.sleep(random.uniform(0.1, 0.5))  # Délai aléatoire
            if response:
                logger.debug(f"Réponse SYN reçue pour {ip}:{port} : {response.summary()}")
                if response.haslayer(TCP) and response[TCP].flags == 0x12:  # SYN-ACK
                    results.update({"status": "open", "confidence": 0.95, "bypass_method": "syn", "protection": "none"})
                    service, confidence = self._predict_service_heuristic(response, response_time)
                    if self.lstm_model:
                        features = self._extract_features(response, response_time)
                        lstm_service, lstm_confidence = self._predict_service_lstm(features)
                        if lstm_confidence > confidence:
                            service, confidence = lstm_service, lstm_confidence
                    results.update({"service": service, "confidence": confidence})
                    os_guess = await self.fingerprint_tcp_ip(response)
                    results["os_guess"] = os_guess
                    logger.info(f"Port {ip}:{port} ouvert via SYN scan : {service}")
                elif response.haslayer(TCP) and response[TCP].flags == 0x14:  # RST
                    results["status"] = "closed"
                    logger.debug(f"Port {ip}:{port} fermé (RST)")
                elif response.haslayer(ICMP) and response[ICMP].type == 3:
                    results.update({"status": "filtered", "protection": "firewall", "confidence": 0.7})
                    logger.debug(f"Port {ip}:{port} filtré (ICMP type 3)")
                else:
                    results["status"] = "filtered"
                    logger.debug(f"Port {ip}:{port} filtré (réponse inattendue)")
            else:
                results["status"] = "filtered"
                self.packet_loss_count += 1
                logger.debug(f"Port {ip}:{port} : aucune réponse SYN")
            return results
        except Exception as e:
            logger.error(f"Erreur lors du scan SYN pour {ip}:{port} : {e}", exc_info=True)
            self.packet_loss_count += 1
            return results

async def grab_banner(self, ip: str, port: int, domain: Optional[str]) -> Dict:
    """Récupère la bannière d'un service."""
    async with self.semaphore:
        results = {"banner": "", "service": "unknown", "inferred_service": "unknown", "confidence": 0.5, "version": ""}
        try:
            async with asyncio.timeout(self.config.TIMEOUT):
                reader, writer = await asyncio.open_connection(ip, port)
                banner = await reader.read(1024)
                banner_str = banner.decode("utf-8", errors="ignore").strip()
                results["banner"] = banner_str
                if "HTTP" in banner_str:
                    results.update({"service": "http", "inferred_service": "http", "confidence": 0.9})
                elif "SSH" in banner_str:
                    results.update({"service": "ssh", "inferred_service": "ssh", "confidence": 0.9})
                writer.close()
                await writer.wait_closed()
                logger.debug(f"Bannière récupérée pour {ip}:{port} : {banner_str[:50]}")
                await asyncio.sleep(random.uniform(0.1, 0.5))  # Délai aléatoire
        except Exception as e:
            logger.debug(f"Échec de la récupération de la bannière pour {ip}:{port} : {e}")
            self.packet_loss_count += 1
        return results

async def identify_service_version(self, ip: str, port: int, banner: str) -> Dict:
    """Identifie la version du service à partir de la bannière."""
    async with self.semaphore:
        results = {"version": "", "confidence": 0.7}
        version_patterns = [
            (r"Server: ([^\r\n]+)", "http"),
            (r"SSH-[\d.]+-([^\s]+)", "ssh"),
            (r"Apache/([\d.]+)", "apache"),
            (r"nginx/([\d.]+)", "nginx"),
            (r"Shopify", "shopify")
        ]
        try:
            for pattern, service_hint in version_patterns:
                match = re.search(pattern, banner, re.IGNORECASE)
                if match:
                    version = match.group(1).strip() if service_hint != "shopify" else "Shopify"
                    results.update({"version": version, "confidence": 0.95})
                    logger.debug(f"Version identifiée pour {ip}:{port} ({service_hint}) : {version}")
                    break
            await asyncio.sleep(random.uniform(0.1, 0.5))  # Délai aléatoire
        except Exception as e:
            logger.error(f"Erreur lors de l'identification de la version pour {ip}:{port} : {e}")
        return results

async def enumerate_web_paths(self, ip: str, port: int, domain: Optional[str] = None) -> List[Dict]:
    """Énumère les chemins web sensibles avec rotation de proxy."""
    async with self.semaphore:
        sensitive_paths = [
            "/admin", "/login", "/cart", "/checkout", "/.env", "/config", "/api"
        ]
        results = []
        target_base = f"http://{domain or ip}:{port}" if port == 80 else f"https://{domain or ip}:{port}"
        try:
            async with asyncio.timeout(self.config.TIMEOUT * 2):
                headers = {
                    "User-Agent": random.choice(self.config.USER_AGENTS),
                    "Host": domain or ip,
                    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
                    "Connection": "keep-alive",
                    "X-Forwarded-For": f"192.168.{random.randint(0, 255)}.{random.randint(1, 255)}"
                }
                ssl_context = ssl.create_default_context(cafile=certifi.where()) if port == 443 else None
                proxy = self._get_next_proxy()
                async with aiohttp.ClientSession(headers=headers, connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
                    tasks = [self._fetch_web_path(session, f"{target_base}{path}", path, ip, port, proxy) for path in sensitive_paths]
                    path_results = await asyncio.gather(*tasks, return_exceptions=True)
                    for res in path_results:
                        if isinstance(res, dict) and res:
                            results.append(res)
                    await asyncio.sleep(random.uniform(0.1, 0.5))  # Délai aléatoire
        except Exception as e:
            logger.error(f"Erreur lors de l'énumération des chemins pour {ip}:{port} : {e}")
            self.packet_loss_count += 1
        return results

async def _fetch_web_path(self, session: aiohttp.ClientSession, url: str, path: str, ip: str, port: int, proxy: Optional[Dict[str, str]]) -> Optional[Dict]:
    """Récupère un chemin web spécifique avec proxy."""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=self.config.TIMEOUT), proxy=proxy["http"] if proxy else None) as response:
            status = response.status
            await asyncio.sleep(random.uniform(0.1, 0.5))  # Délai aléatoire
            if status == 200:
                logger.debug(f"Chemin web {url} accessible (status 200)")
                return {"path": path, "status_code": status, "confidence": 0.95}
            elif status in [301, 302, 307, 308]:
                logger.debug(f"Chemin web {url} redirigé (status {status})")
                return {"path": path, "status_code": status, "confidence": 0.9, "redirect_to": response.headers.get('Location', '')}
            elif status == 403:
                logger.debug(f"Chemin web {url} bloqué (status 403, possible WAF)")
                return {"path": path, "status_code": status, "confidence": 0.85}
    except Exception as e:
        logger.debug(f"Erreur lors de la récupération du chemin {url} avec proxy {proxy}: {e}")
        self.packet_loss_count += 1
    return None

async def advanced_ssl_analysis(self, ip: str, port: int, domain: Optional[str] = None) -> Dict:
    """Analyse avancée des certificats SSL."""
    async with self.semaphore:
        results = {"issuer": "", "san_domains": [], "not_after": "", "serial_number": "", "confidence": 0.0, "error": "", "protocols": [], "cipher": ""}
        hostname = domain if domain and domain != ip else None
        try:
            ssl_results = await asyncio.get_event_loop().run_in_executor(
                self.state.executor, self._sync_advanced_ssl_analysis, ip, port, hostname
            )
            results.update(ssl_results)
            await asyncio.sleep(random.uniform(0.1, 0.5))  # Délai aléatoire
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse SSL pour {ip}:{port} : {e}")
            self.packet_loss_count += 1
        return results

def _sync_advanced_ssl_analysis(self, ip: str, port: int, hostname: Optional[str]) -> Dict:
    """Analyse synchrone des certificats SSL."""
    results = {"issuer": "", "san_domains": [], "not_after": "", "serial_number": "", "confidence": 0.0, "error": "", "protocols": [], "cipher": ""}
    context = ssl.create_default_context(cafile=certifi.where())
    context.check_hostname = False
    context.verify_mode = ssl.CERT_OPTIONAL
    try:
        with socket.create_connection((ip, port), timeout=self.config.TIMEOUT) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                cert = ssock.getpeercert()
                if cert:
                    issuer_info = dict(x[0] for x in cert.get("issuer", []))
                    results.update({
                        "issuer": issuer_info.get("commonName", issuer_info.get("organizationName", "Unknown")),
                        "san_domains": [val for type_val, val in cert.get("subjectAltName", []) if type_val.lower() == 'dns'],
                        "not_after": cert.get("notAfter", ""),
                        "serial_number": cert.get("serialNumber", ""),
                        "protocols": [ssock.version()] if ssock.version() else [],
                        "cipher": ssock.cipher()[0] if ssock.cipher() else "",
                        "confidence": 0.95
                    })
                    logger.debug(f"Analyse SSL réussie pour {ip}:{port}")
    except Exception as e:
        results.update({"error": str(e), "confidence": 0.5})
        logger.error(f"Erreur SSL pour {ip}:{port} : {e}")
        self.packet_loss_count += 1
    return results

async def check_vulnerabilities(self, ip: str, port: int, version: str, banner: str) -> List[Dict]:
    """Vérifie les vulnérabilités connues."""
    async with self.semaphore:
        results = []
        known_vulnerabilities = {
            "apache": [
                (re.compile(r"^2\.2\..*"), "Apache 2.2.x obsolète (multiples CVEs)", "critical"),
                (re.compile(r"^2\.4\.([0-6]|\d{2,})$"), "Apache 2.4.x ancien (vérifiez CVE-2021-42013)", "high"),
            ],
            "nginx": [
                (re.compile(r"^1\.(1[0-9]|2[0-1])\..*"), "Nginx < 1.22.0 vulnérable", "medium"),
            ],
            "shopify": [
                (re.compile(r".*"), "Vérifiez les configurations WAF Cloudflare", "medium")
            ]
        }
        try:
            service = ""
            if "apache" in version.lower():
                service = "apache"
            elif "nginx" in version.lower():
                service = "nginx"
            elif "shopify" in version.lower():
                service = "shopify"
            if service in known_vulnerabilities:
                for pattern, desc, severity in known_vulnerabilities[service]:
                    if pattern.match(version):
                        results.append({"description": desc, "severity": severity, "confidence": 0.95})
                        logger.debug(f"Vulnérabilité détectée pour {ip}:{port} : {desc}")
            await asyncio.sleep(random.uniform(0.1, 0.5))  # Délai aléatoire
        except Exception as e:
            logger.error(f"Erreur lors de la vérification des vulnérabilités pour {ip}:{port} : {e}")
        return results

async def advanced_subdomain_enumeration(self, ip: str, domain: Optional[str]) -> List[str]:
    """Énumère les sous-domaines associés à une IP."""
    async with self.semaphore:
        subdomains = []
        if not domain or re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", domain):
            logger.debug(f"Énumération des sous-domaines ignorée pour {ip} : aucun domaine valide fourni")
            return subdomains
        common_subdomains = ["www", "mail", "api", "admin", "test", "shop", "store"]
        tasks = [self._check_subdomain_resolves_to_ip(f"{sub}.{domain}", ip) for sub in common_subdomains]
        try:
            async with asyncio.timeout(self.config.TIMEOUT * 3):
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for res in results:
                    if isinstance(res, str):
                        subdomains.append(res)
                        logger.debug(f"Sous-domaine trouvé : {res}")
                await asyncio.sleep(random.uniform(0.1, 0.5))  # Délai aléatoire
        except Exception as e:
            logger.error(f"Erreur lors de l'énumération des sous-domaines pour {domain} : {e}")
            self.packet_loss_count += 1
        return list(set(subdomains))

async def _check_subdomain_resolves_to_ip(self, subdomain: str, target_ip: str) -> Optional[str]:
    """Vérifie si un sous-domaine résout vers l'IP cible."""
    try:
        answers = await asyncio.get_event_loop().run_in_executor(None, lambda: dns.resolver.resolve(subdomain, 'A'))
        for rdata in answers:
            if rdata.address == target_ip:
                logger.debug(f"Sous-domaine {subdomain} résout vers {target_ip}")
                return subdomain
        await asyncio.sleep(random.uniform(0.1, 0.5))  # Délai aléatoire
    except Exception as e:
        logger.debug(f"Erreur lors de la résolution de {subdomain} : {e}")
    return None

async def waf_bypass(self, ip: str, port: int, domain: Optional[str] = None) -> Dict:
    """Tente de contourner les WAF avec rotation de proxy."""
    async with self.semaphore:
        results = {"bypass_success": False, "waf_detected": "none", "confidence": 0.0, "bypass_method": "none"}
        target_url = f"http://{domain or ip}:{port}" if port == 80 else f"https://{domain or ip}:{port}"
        payloads = [
            {
                "headers": {
                    "User-Agent": random.choice(self.config.USER_AGENTS),
                    "X-Forwarded-For": f"192.168.{random.randint(0, 255)}.{random.randint(1, 255)}",
                    "Host": domain or ip,
                    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
                    "Connection": "keep-alive"
                },
                "url_param": "?bypass=true",
                "method": "GET"
            },
            {
                "headers": {
                    "User-Agent": "Googlebot/2.1 (+http://www.google.com/bot.html)",
                    "Host": domain or ip,
                    "Accept": "text/html",
                    "Connection": "keep-alive"
                },
                "url_param": "",
                "method": "HEAD"
            },
            {
                "headers": {
                    "User-Agent": random.choice(self.config.USER_AGENTS),
                    "Host": domain or ip,
                    "Origin": f"https://{domain or ip}",
                    "Referer": f"https://{domain or ip}/",
                    "X-Forwarded-For": f"192.168.{random.randint(0, 255)}.{random.randint(1, 255)}"
                },
                "url_param": "/?rand=" + str(random.randint(1000, 9999)),
                "method": "GET"
            }
        ]

        for attempt, payload_info in enumerate(payloads):
            current_headers = payload_info["headers"]
            current_url = target_url + payload_info.get("url_param", "")
            method = payload_info["method"]
            proxy = self._get_next_proxy()
            try:
                async with asyncio.timeout(self.config.TIMEOUT):
                    ssl_context = ssl.create_default_context(cafile=certifi.where()) if port == 443 else None
                    async with aiohttp.ClientSession(headers=current_headers, connector=aiohttp.TCPConnector(ssl=ssl_context, limit_per_host=1)) as session:
                        logger.debug(f"WAF bypass: Tentative {attempt+1}/{len(payloads)} pour {current_url} avec méthode {method}, proxy {proxy}")
                        async with session.request(method, current_url, proxy=proxy["http"] if proxy else None) as response:
                            status = response.status
                            response_text = await response.text(errors='ignore')
                            server_header = response.headers.get("Server", "").lower()
                            waf_keywords = ["cloudflare", "akamai", "sucuri", "incapsula", "barracuda", "f5", "mod_security"]
                            response_text_lower = response_text.lower()

                            detected_waf = "none"
                            for kw in waf_keywords:
                                if kw in server_header or kw in response_text_lower or kw in str(response.headers).lower():
                                    detected_waf = kw
                                    break

                            if status == 200:
                                results.update({
                                    "bypass_success": True,
                                    "confidence": 0.95,
                                    "waf_detected": detected_waf if detected_waf != "none" else "bypassed_unknown",
                                    "bypass_method": f"payload_{attempt+1}"
                                })
                                logger.info(f"WAF bypass: Succès pour {current_url} (status 200), WAF détecté: {results['waf_detected']}")
                                return results
                            elif status in [403, 429, 503]:
                                results.update({
                                    "waf_detected": detected_waf if detected_waf != "none" else "generic_block",
                                    "confidence": 0.85,
                                    "bypass_success": False
                                })
                                logger.info(f"WAF bypass: Bloqué pour {current_url} (status {status}), WAF détecté: {results['waf_detected']}")
                            else:
                                results.update({"confidence": 0.80, "bypass_success": False, "waf_detected": detected_waf})
                                logger.debug(f"WAF bypass: Status {status} pour {current_url}, WAF: {results['waf_detected']}")
                        await asyncio.sleep(random.uniform(0.1, 0.5))  # Délai aléatoire
            except asyncio.TimeoutError:
                logger.debug(f"WAF bypass: Timeout pour {current_url} (tentative {attempt+1})")
                self.packet_loss_count += 1
            except aiohttp.ClientError as e:
                logger.debug(f"WAF bypass: ClientError pour {current_url} (tentative {attempt+1}) : {e}")
                self.packet_loss_count += 1
            if attempt < len(payloads) - 1:
                await asyncio.sleep(random.uniform(0.1, 0.5))  # Délai aléatoire
        logger.info(f"WAF bypass: Toutes les {len(payloads)} tentatives ont échoué pour {target_url}")
        return results

async def cloudflare_stealth_probe(self, ip: str, port: int, domain: Optional[str] = None) -> Dict:
    """Probe stealth pour détecter et contourner Cloudflare avec rotation de proxy."""
    async with self.semaphore:
        results = {"service": "unknown", "version": "", "banner": "", "confidence": 0.0, "bypass_success": False, "protection": "none"}
        target_url = f"http://{domain or ip}:{port}" if port == 80 else f"https://{domain or ip}:{port}"
        user_agents = self.config.USER_AGENTS + ["ShopifyBot/1.0 (+https://shopify.com)"]
        for attempt in range(self.config.MAX_RETRIES):
            try:
                async with asyncio.timeout(self.config.TIMEOUT):
                    headers = {
                        "User-Agent": random.choice(user_agents),
                        "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.9",
                        "Connection": "keep-alive",
                        "Host": domain or ip,
                        "Referer": f"https://{domain or ip}/",
                        "DNT": "1",
                        "Upgrade-Insecure-Requests": "1",
                        "X-Forwarded-For": f"192.168.{random.randint(0, 255)}.{random.randint(1, 255)}"
                    }
                    ssl_context = ssl.create_default_context(cafile=certifi.where()) if port == 443 else None
                    proxy = self._get_next_proxy()
                    async with aiohttp.ClientSession(headers=headers, connector=aiohttp.TCPConnector(ssl=ssl_context, limit_per_host=1)) as session:
                        logger.debug(f"Cloudflare stealth probe: Tentative {attempt+1} pour {target_url} avec proxy {proxy}")
                        async with session.get(target_url, proxy=proxy["http"] if proxy else None) as response:
                            banner = await response.text(errors='ignore')
                            results["banner"] = banner[:200]
                            server_header = response.headers.get("Server", "").lower()
                            is_cloudflare = "cloudflare" in server_header or "cf-ray" in response.headers
                            is_shopify = "shopify" in banner.lower()

                            if response.status == 200:
                                results.update({
                                    "service": "http" if port == 80 else "https",
                                    "version": "Shopify" if is_shopify else "",
                                    "confidence": 0.95,
                                    "bypass_success": True,
                                    "protection": "cloudflare" if is_cloudflare else "unknown"
                                })
                                logger.info(f"Cloudflare stealth probe: Succès pour {target_url} (status 200)")
                                return results
                            elif response.status in [403, 503]:
                                results.update({
                                    "protection": "cloudflare" if is_cloudflare else "blocked",
                                    "confidence": 0.90,
                                    "bypass_success": False
                                })
                                logger.info(f"Cloudflare stealth probe: Bloqué pour {target_url} (status {response.status})")
                            else:
                                results.update({"confidence": 0.80, "bypass_success": False, "protection": "unknown"})
                                logger.debug(f"Cloudflare stealth probe: Status {response.status} pour {target_url}")
                        await asyncio.sleep(random.uniform(0.1, 0.5))  # Délai aléatoire
            except Exception as e:
                logger.debug(f"Cloudflare stealth probe: Erreur pour {target_url} (tentative {attempt+1}) : {e}")
                self.packet_loss_count += 1
                if attempt < self.config.MAX_RETRIES - 1:
                    await asyncio.sleep(random.uniform(0.1, 0.5))  # Délai aléatoire
        results.update({"confidence": 0.60, "protection": "error"})
        return results

async def cloudflare_js_challenge_solver(self, ip: str, port: int, domain: Optional[str] = None) -> Dict:
    """Résout les défis JavaScript de Cloudflare."""
    async with self.semaphore:
        results = {"bypass_success": False, "confidence": 0.0, "bypass_method": "js_challenge"}
        try:
            from webdriver_manager.chrome import ChromeDriverManager
            from selenium import webdriver as selenium_webdriver
            from selenium.webdriver.chrome.options import Options as SeleniumOptions
            from selenium.webdriver.chrome.service import Service
            from selenium.common.exceptions import TimeoutException as SeleniumTimeoutException
            from selenium.common.exceptions import WebDriverException as SeleniumWebDriverException
        except ImportError:
            logger.error("JS challenge solver: 'webdriver-manager' ou 'selenium' non installé.")
            results.update({"confidence": 0.10, "error": "webdriver-manager or selenium not installed"})
            return results

        if not shutil.which("chromedriver"):
            logger.error("JS challenge solver: chromedriver non trouvé dans PATH.")
            results.update({"confidence": 0.10, "error": "chromedriver not found"})
            return results

        target_url = f"http://{domain or ip}:{port}" if port == 80 else f"https://{domain or ip}:{port}"
        try:
            async with asyncio.timeout(self.config.TIMEOUT * 5):
                selenium_results = await asyncio.get_event_loop().run_in_executor(
                    self.state.executor, lambda: self._sync_selenium_js_challenge(target_url)
                )
                results.update(selenium_results)
                await asyncio.sleep(random.uniform(0.1, 0.5))  # Délai aléatoire
        except asyncio.TimeoutError:
            logger.debug(f"JS challenge solver: Timeout pour {target_url}")
            results.update({"confidence": 0.75, "bypass_success": False, "error": "Overall operation timeout"})
            self.packet_loss_count += 1
        except Exception as e:
            logger.error(f"JS challenge solver: Erreur pour {target_url}: {e}", exc_info=True)
            results.update({"confidence": 0.40, "bypass_success": False, "error": f"Unexpected executor error: {e}"})
        return results

def _sync_selenium_js_challenge(self, target_url: str) -> Dict:
    """Helper synchrone pour résoudre les défis JavaScript via Selenium."""
    results = {"bypass_success": False, "confidence": 0.0}
    driver = None
    try:
        from webdriver_manager.chrome import ChromeDriverManager
        from selenium.webdriver.chrome.webdriver import WebDriver as selenium_webdriver
        from selenium.webdriver.chrome.options import Options as SeleniumOptions
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.common.exceptions import TimeoutException as SeleniumTimeoutException
        from selenium.common.exceptions import WebDriverException as SeleniumWebDriverException
    except ImportError:
        logger.error("JS challenge solver: 'webdriver-manager' ou 'selenium' non installé.")
        return {"success": False, "confidence": 0.0, "message": "Module 'webdriver-manager' ou 'selenium' manquant."}

    try:
        chrome_options = SeleniumOptions()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument(f"--user-agent={random.choice(self.config.USER_AGENTS)}")

        proxy = self._get_next_proxy()
        if proxy:
            chrome_options.add_argument(f"--proxy-server={proxy['http']}")

        service = Service(ChromeDriverManager(log_level=logging.WARNING).install())
        driver = selenium_webdriver.Chrome(service=service, options=chrome_options)
        driver.set_page_load_timeout(self.config.TIMEOUT * 2)

        logger.debug(f"Sync JS challenge: Navigation vers {target_url}")
        driver.get(target_url)
        time.sleep(7)

        page_title = driver.title.lower()
        page_source_lower = driver.page_source.lower()

        if "just a moment" in page_title or "checking your browser" in page_title or "challenge" in page_source_lower:
            logger.info(f"Sync JS challenge: Défi détecté pour {target_url}. Attente.")
            time.sleep(10)
            page_title = driver.title.lower()
            page_source_lower = driver.page_source.lower()

            if "just a moment" in page_title or "checking your browser" in page_title or "challenge" in page_source_lower:
                results.update({"confidence": 0.85, "bypass_success": False, "error": "Challenge page persistent"})
                logger.info(f"Sync JS challenge: Échec pour {target_url} (page de défi persistante).")
            else:
                results.update({"bypass_success": True, "confidence": 0.95})
                logger.info(f"Sync JS challenge: Succès ou pas de défi pour {target_url}.")
    except SeleniumTimeoutException as e:
        logger.debug(f"Sync JS challenge: Timeout pour {target_url}: {e.msg}")
        results.update({"confidence": 0.70, "bypass_success": False, "error": f"Selenium timeout: {e.msg}"})
        self.packet_loss_count += 1
    except Exception as e:
        logger.error(f"Sync JS challenge: Erreur pour {target_url}: {e}", exc_info=True)
        results.update({"confidence": 0.50, "bypass_success": False, "error": f"Unexpected sync error: {e}"})
    finally:
        if driver:
            try:
                driver.quit()
            except Exception as e_quit:
                logger.warning(f"Sync JS challenge: Erreur lors de la fermeture du WebDriver: {e_quit}")
    return results

async def cloudflare_rate_limit_evasion(self, ip: str, port: int, domain: Optional[str] = None) -> Dict:
    """Évite les limites de débit de Cloudflare avec rotation de proxy."""
    async with self.semaphore:
        results = {"evasion_success": False, "confidence": 0.0}
        target_url = f"http://{domain or ip}:{port}" if port == 80 else f"https://{domain or ip}:{port}"
        user_agents = self.config.USER_AGENTS + ["ShopifyBot/1.0 (+https://shopify.com)"]
        try:
            async with asyncio.timeout(self.config.TIMEOUT * self.config.MAX_RETRIES):
                for attempt in range(self.config.MAX_RETRIES):
                    headers = {
                        "User-Agent": random.choice(user_agents),
                        "X-Forwarded-For": f"192.168.{random.randint(0, 255)}.{random.randint(1, 255)}"
                    }
                    ssl_context = ssl.create_default_context(cafile=certifi.where()) if port == 443 else None
                    proxy = self._get_next_proxy()
                    async with aiohttp.ClientSession(headers=headers, connector=aiohttp.TCPConnector(ssl=ssl_context, limit_per_host=1)) as session:
                        try:
                            await asyncio.sleep(random.uniform(0.1, 0.5))  # Délai aléatoire
                            logger.debug(f"Cloudflare rate limit evasion: Tentative {attempt+1} pour {target_url} avec proxy {proxy}")
                            async with session.get(target_url, proxy=proxy["http"] if proxy else None, timeout=aiohttp.ClientTimeout(total=self.config.TIMEOUT)) as response:
                                if response.status != 429:
                                    results.update({"evasion_success": True, "confidence": 0.90})
                                    logger.info(f"Cloudflare rate limit evasion: Succès pour {target_url} (status {response.status})")
                                    return results
                                else:
                                    logger.debug(f"Cloudflare rate limit evasion: Tentative {attempt+1} pour {target_url} hit 429.")
                        except asyncio.TimeoutError:
                            logger.debug(f"Cloudflare rate limit evasion: Timeout pour {target_url} (tentative {attempt+1})")
                            self.packet_loss_count += 1
                        except aiohttp.ClientError as e:
                            logger.debug(f"Cloudflare rate limit evasion: ClientError pour {target_url} (tentative {attempt+1}) : {e}")
                            self.packet_loss_count += 1
        except asyncio.TimeoutError:
            logger.debug(f"Cloudflare rate limit evasion: Timeout global pour {target_url}")
            results.update({"confidence": 0.75, "evasion_success": False})
            self.packet_loss_count += 1
        return results

async def cloudflare_tcp_fingerprint(self, ip: str, port: int) -> Dict:
    """Analyse TCP fingerprint pour détecter Cloudflare avec rotation d'IP source."""
    async with self.semaphore:
        results = {"inferred_protection": "none", "confidence": 0.0}
        if not self.state.is_admin:
            logger.warning(f"Cloudflare TCP fingerprint: Désactivé pour {ip}:{port}, privilèges requis.")
            return results
        try:
            async with asyncio.timeout(self.config.TIMEOUT):
                tcp_options = [("MSS", 1460), ("NOP", None), ("WScale", 1), ("SACKPermitted", None)]
                source_ip = self._get_random_source_ip()
                pkt = IP(dst=ip, src=source_ip) / TCP(dport=port, flags="S", options=tcp_options) if source_ip else IP(dst=ip) / TCP(dport=port, flags="S", options=tcp_options)
                start_time = time.time()
                response = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: sr1(pkt, timeout=self.config.TIMEOUT, verbose=0)
                )
                response_time = time.time() - start_time
                self.response_times.append(response_time)
                await asyncio.sleep(random.uniform(0.1, 0.5))  # Délai aléatoire
                if response and response.haslayer(TCP):
                    ttl = response[IP].ttl
                    if 110 <= ttl <= 120:
                        results.update({"inferred_protection": "cloudflare", "confidence": 0.85})
                        logger.debug(f"Cloudflare TCP fingerprint: TTL {ttl} détecté pour {ip}:{port}")
                    else:
                        results.update({"inferred_protection": "none", "confidence": 0.80})
                else:
                    results.update({"inferred_protection": "unknown", "confidence": 0.75})
                    self.packet_loss_count += 1
                    logger.debug(f"Cloudflare TCP fingerprint: Aucune réponse TCP pour {ip}:{port}")
        except Exception as e:
            logger.error(f"Erreur lors de la fingerprint TCP pour {ip}:{port}: {e}", exc_info=True)
            results.update({"inferred_protection": "unknown", "confidence": 0.70})
            self.packet_loss_count += 1
        return results

async def _network_noise_analysis(self, ip: str, port: int) -> Dict:
    """Analyse les variations de délais sous charge pour détecter IDS/IPS avec rotation d'IP source."""
    async with self.semaphore:
        results = {"protection": "none", "protection_confidence": 0.5}
        try:
            response_times = []
            for _ in range(5):
                source_ip = self._get_random_source_ip()
                pkt = IP(dst=ip, src=source_ip) / TCP(dport=port, flags="S", sport=RandShort()) if source_ip else IP(dst=ip) / TCP(dport=port, flags="S", sport=RandShort())
                start_time = time.time()
                response = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: sr1(pkt, timeout=self.config.TIMEOUT / 2, verbose=0)
                )
                response_time = time.time() - start_time
                response_times.append(response_time)
                await asyncio.sleep(random.uniform(0.1, 0.5))  # Délai aléatoire
                if not response:
                    self.packet_loss_count += 1

            if response_times:
                avg_time = sum(response_times) / len(response_times)
                std_dev = (sum((t - avg_time) ** 2 for t in response_times) / len(response_times)) ** 0.5
                if std_dev > 0.1 * avg_time:
                    results.update({"protection": "ids_ips", "protection_confidence": 0.8})
                    logger.debug(f"Protection IDS/IPS détectée sur {ip}:{port} : Écart-type {std_dev:.3f}s vs moyenne {avg_time:.3f}s")
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse de bruit réseau sur {ip}:{port} : {e}", exc_info=True)
            self.packet_loss_count += 1
        return results

async def detect_protection_advanced(self, ip: str, port: int) -> Dict:
    """Détecte les protections avancées via analyse des délais et paquets anormaux avec rotation d'IP source."""
    async with self.semaphore:
        results = {"protection": "none", "protection_confidence": 0.5}
        try:
            source_ip = self._get_random_source_ip()
            pkt = IP(dst=ip, src=source_ip) / TCP(dport=port, flags="A", sport=RandShort(), seq=random.randint(1000, 1000000)) if source_ip else IP(dst=ip) / TCP(dport=port, flags="A", sport=RandShort(), seq=random.randint(1000, 1000000))
            start_time = time.time()
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: sr1(pkt, timeout=self.config.TIMEOUT, verbose=0)
            )
            response_time = time.time() - start_time
            self.response_times.append(response_time)
            await asyncio.sleep(random.uniform(0.1, 0.5))  # Délai aléatoire

            if response:
                if response.haslayer(ICMP) and response[ICMP].type == 3:
                    results.update({"protection": "firewall", "protection_confidence": 0.8})
                    logger.debug(f"Protection détectée sur {ip}:{port} : ICMP type 3 (pare-feu)")
                elif response.haslayer(TCP) and response[TCP].flags == "R":
                    avg_response_time = sum(self.response_times[-5:]) / min(len(self.response_times), 5) if self.response_times else response_time
                    if response_time > 2 * avg_response_time:
                        results.update({"protection": "ids_ips", "protection_confidence": 0.75})
                        logger.debug(f"Protection IDS/IPS détectée sur {ip}:{port} : Délai {response_time:.3f}s vs moyenne {avg_response_time:.3f}s")
                    else:
                        results.update({"protection": "firewall", "protection_confidence": 0.7})
                        logger.debug(f"Protection détectée sur {ip}:{port} : Réponse inattendue (pare-feu)")
                else:
                    results.update({"protection": "firewall", "protection_confidence": 0.7})
                    logger.debug(f"Protection détectée sur {ip}:{port} : Réponse inattendue (pare-feu)")
            else:
                results.update({"protection": "none", "protection_confidence": 0.6})
                self.packet_loss_count += 1
                logger.debug(f"Aucune protection détectée sur {ip}:{port} : Pas de réponse")
        except Exception as e:
            logger.error(f"Erreur lors de la détection de protection sur {ip}:{port} : {e}", exc_info=True)
            self.packet_loss_count += 1
        return results

async def do_full_scan_on_port(self, ip: str, port: int, domain: Optional[str] = None, zombie_ip: Optional[str] = None) -> Dict:
    """Effectue un scan complet sur un port."""
    async with self.semaphore:
        results = {
            "ip": ip, "port": port, "status": "error", "service": "unknown", "inferred_service": "unknown",
            "inferred_state": "unknown", "banner": "", "version": "", "confidence": 0.1,
            "protection": "unknown", "protection_confidence": 0.0, "bypass_method": "none", "vulnerabilities": [],
            "sensitive_paths": [], "ssl_details": {}, "subdomains": [], "waf_bypass_info": {}, "cloudflare_info": {},
            "os_guess": "unknown"
        }
        try:
            logger.info(f"Démarrage du scan complet pour {ip}:{port}")
            # Vérification si l'IP est issue d'une résolution DNS
            if domain and not re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", domain):
                resolved_ips = await self.resolve_domain(domain)
                if ip not in resolved_ips:
                    logger.warning(f"L'IP {ip} ne correspond pas aux IPs résolues pour {domain}: {resolved_ips}")

            tcp_results = await self.tcp_connect(ip, port)
            results.update({
                "status": tcp_results["status"],
                "service": tcp_results["service"],
                "confidence": tcp_results["confidence"],
                "protection": tcp_results["protection"],
                "protection_confidence": tcp_results.get("protection_confidence", 0.5),
                "os_guess": tcp_results["os_guess"]
            })
            logger.info(f"Résultat TCP connect pour {ip}:{port} : {tcp_results}")

            # Toujours tenter SYN scan si admin, sinon fallback déjà géré dans syn_scan
            syn_results = await self.syn_scan(ip, port)
            logger.info(f"Résultat SYN scan pour {ip}:{port} : {syn_results}")
            if syn_results["status"] == "open" and results["status"] != "open":
                results.update({
                    "status": syn_results["status"],
                    "service": syn_results["service"],
                    "confidence": syn_results["confidence"],
                    "bypass_method": syn_results["bypass_method"],
                    "protection": syn_results["protection"],
                    "protection_confidence": syn_results.get("protection_confidence", 0.5),
                    "os_guess": syn_results["os_guess"]
                })

            if results["status"] == "open":
                banner_results = await self.grab_banner(ip, port, domain)
                logger.info(f"Bannière pour {ip}:{port} : {banner_results}")
                results.update({
                    "banner": banner_results["banner"],
                    "service": banner_results["service"],
                    "inferred_service": banner_results["inferred_service"],
                    "confidence": max(results["confidence"], banner_results["confidence"]),
                    "version": banner_results["version"]
                })
                version_results = await self.identify_service_version(ip, port, banner_results["banner"])
                results["version"] = version_results["version"] or results["version"]
                results["confidence"] = max(results["confidence"], version_results["confidence"])
                vuln_results = await self.check_vulnerabilities(ip, port, results["version"], banner_results["banner"])
                results["vulnerabilities"] = vuln_results

            if port in [80, 443, 8080, 8443]:
                waf_results = await self.waf_bypass(ip, port, domain)
                results["waf_bypass_info"] = waf_results
                logger.info(f"Résultat WAF bypass pour {ip}:{port} : {waf_results}")
                if waf_results["bypass_success"]:
                    results["protection"] = waf_results["waf_detected"]
                    results["protection_confidence"] = waf_results["confidence"]
                cf_results = await self.cloudflare_stealth_probe(ip, port, domain)
                results["cloudflare_info"] = cf_results
                logger.info(f"Résultat Cloudflare probe pour {ip}:{port} : {cf_results}")
                if cf_results["bypass_success"]:
                    results["protection"] = cf_results["protection"]
                    results["protection_confidence"] = cf_results["confidence"]
                else:
                    js_results = await self.cloudflare_js_challenge_solver(ip, port, domain)
                    results["cloudflare_info"].update({"js_challenge": js_results})
                    logger.info(f"Résultat JS challenge pour {ip}:{port} : {js_results}")
                    rate_limit_results = await self.cloudflare_rate_limit_evasion(ip, port, domain)
                    results["cloudflare_info"].update({"rate_limit_evasion": rate_limit_results})
                    logger.info(f"Résultat rate limit evasion pour {ip}:{port} : {rate_limit_results}")
                path_results = await self.enumerate_web_paths(ip, port, domain)
                results["sensitive_paths"] = path_results
                if port == 443 or results["service"] == "https":
                    ssl_results = await self.advanced_ssl_analysis(ip, port, domain)
                    results["ssl_details"] = ssl_results
                    logger.info(f"Résultat SSL analysis pour {ip}:{port} : {ssl_results}")

            if domain and not re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", domain):
                subdomain_results = await self.advanced_subdomain_enumeration(ip, domain)
                results["subdomains"] = subdomain_results
                logger.info(f"Sous-domaines pour {ip}:{port} : {subdomain_results}")

            if zombie_ip:
                logger.warning(f"Scan zombie non implémenté pour {ip}:{port}")

            # Log des pertes de paquets
            if self.packet_loss_count > 0:
                logger.warning(f"Pertes de paquets détectées lors du scan de {ip}:{port} : {self.packet_loss_count} pertes")
        except Exception as e:
            logger.error(f"Erreur lors du scan complet de {ip}:{port} : {e}", exc_info=True)
        return results

async def scan_all_ports(self, ip: str, domain: Optional[str] = None) -> List[Dict]:
    """Scanne tous les ports (1 à 65535)."""
    tasks = [self.do_full_scan_on_port(ip, port, domain) for port in range(1, 65536)]
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
    result = await scanner.do_full_scan_on_port("8.8.8.8", 53, domain=None)  # Test avec IP directe
    print(result)

if __name__ == "__main__":
    asyncio.run(main())