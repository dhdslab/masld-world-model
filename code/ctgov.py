from __future__ import annotations

import requests

UA = {"User-Agent": "MASLD-evidence-map-count-check/1.0 (research script; no contact)"}
ADV_COND = ('cirrhosis OR "hepatocellular carcinoma" OR "liver cancer" OR "portal hypertension" OR ascites OR '
            '"hepatic encephalopathy" OR "esophageal varices" OR "oesophageal varices" OR "liver transplantation" OR '
            '"liver failure" OR "decompensated liver"')
MASLD_TERM = ('NAFLD OR NASH OR MASLD OR MASH OR MAFLD OR MetALD OR "fatty liver" OR steatohepatitis OR '
              '"steatotic liver" OR "metabolic dysfunction-associated"')


def _session():
    import ssl
    import truststore
    from requests.adapters import HTTPAdapter

    class TrustStoreAdapter(HTTPAdapter):
        def init_poolmanager(self, *a, **kw):
            kw["ssl_context"] = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            return super().init_poolmanager(*a, **kw)
    s = requests.Session(); s.mount("https://", TrustStoreAdapter()); s.headers.update(UA)
    return s


SESSION = None


def get(url, params):
    global SESSION
    SESSION = SESSION or _session()
    return SESSION.get(url, params=params, timeout=60)
