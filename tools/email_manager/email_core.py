"""
Email Manager Core - Gerencia múltiplas contas de email temporário usando Mail.tm API
API Docs: https://docs.mail.tm/
"""

import requests
import json
import random
import string
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
import time


@dataclass
class EmailAccount:
    """Representa uma conta de email"""
    id: str
    address: str
    password: str
    token: str
    created_at: str
    last_checked: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'EmailAccount':
        return cls(**data)


@dataclass
class EmailMessage:
    """Representa uma mensagem de email"""
    id: str
    from_address: str
    from_name: str
    subject: str
    intro: str
    text: str
    html: str
    created_at: str
    has_attachments: bool
    
    
class MailTMClient:
    """Cliente para a API Mail.tm - 100% gratuito, sem limites"""
    
    BASE_URL = "https://api.mail.tm"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
    
    def get_domains(self) -> List[str]:
        """Retorna lista de domínios disponíveis"""
        response = self.session.get(f"{self.BASE_URL}/domains")
        response.raise_for_status()
        data = response.json()
        
        # API pode retornar lista direta ou dict com hydra:member
        if isinstance(data, list):
            return [d["domain"] for d in data]
        elif isinstance(data, dict) and "hydra:member" in data:
            return [d["domain"] for d in data["hydra:member"]]
        else:
            return []
    
    def create_account(self, address: Optional[str] = None, password: Optional[str] = None) -> EmailAccount:
        """Cria uma nova conta de email"""
        domains = self.get_domains()
        if not domains:
            raise Exception("Nenhum domínio disponível")
        
        domain = random.choice(domains)
        
        if not address:
            username = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
            address = f"{username}@{domain}"
        elif "@" not in address:
            address = f"{address}@{domain}"
            
        if not password:
            password = ''.join(random.choices(string.ascii_letters + string.digits + "!@#$%", k=16))
        
        payload = {
            "address": address,
            "password": password
        }
        
        response = self.session.post(f"{self.BASE_URL}/accounts", json=payload)
        response.raise_for_status()
        account_data = response.json()
        
        token = self.get_token(address, password)
        
        return EmailAccount(
            id=account_data["id"],
            address=address,
            password=password,
            token=token,
            created_at=datetime.now().isoformat()
        )
    
    def get_token(self, address: str, password: str) -> str:
        """Obtém token de autenticação"""
        payload = {
            "address": address,
            "password": password
        }
        response = self.session.post(f"{self.BASE_URL}/token", json=payload)
        response.raise_for_status()
        return response.json()["token"]
    
    def get_messages(self, token: str) -> List[EmailMessage]:
        """Retorna todas as mensagens da conta"""
        headers = {"Authorization": f"Bearer {token}"}
        response = self.session.get(f"{self.BASE_URL}/messages", headers=headers)
        response.raise_for_status()
        
        data = response.json()
        
        # API pode retornar lista direta ou dict com hydra:member
        if isinstance(data, list):
            msg_list = data
        elif isinstance(data, dict) and "hydra:member" in data:
            msg_list = data["hydra:member"]
        else:
            msg_list = []
        
        messages = []
        for msg in msg_list:
            messages.append(EmailMessage(
                id=msg["id"],
                from_address=msg["from"]["address"],
                from_name=msg["from"].get("name", ""),
                subject=msg["subject"],
                intro=msg.get("intro", ""),
                text="",
                html="",
                created_at=msg["createdAt"],
                has_attachments=msg["hasAttachments"]
            ))
        return messages
    
    def get_message(self, token: str, message_id: str) -> EmailMessage:
        """Retorna detalhes completos de uma mensagem"""
        headers = {"Authorization": f"Bearer {token}"}
        response = self.session.get(f"{self.BASE_URL}/messages/{message_id}", headers=headers)
        response.raise_for_status()
        msg = response.json()
        
        return EmailMessage(
            id=msg["id"],
            from_address=msg["from"]["address"],
            from_name=msg["from"].get("name", ""),
            subject=msg["subject"],
            intro=msg.get("intro", ""),
            text=msg.get("text", ""),
            html=msg.get("html", [""])[0] if isinstance(msg.get("html"), list) else msg.get("html", ""),
            created_at=msg["createdAt"],
            has_attachments=msg["hasAttachments"]
        )
    
    def delete_message(self, token: str, message_id: str) -> bool:
        """Deleta uma mensagem"""
        headers = {"Authorization": f"Bearer {token}"}
        response = self.session.delete(f"{self.BASE_URL}/messages/{message_id}", headers=headers)
        return response.status_code == 204
    
    def delete_account(self, token: str, account_id: str) -> bool:
        """Deleta uma conta"""
        headers = {"Authorization": f"Bearer {token}"}
        response = self.session.delete(f"{self.BASE_URL}/accounts/{account_id}", headers=headers)
        return response.status_code == 204


class EmailManager:
    """Gerenciador de múltiplas contas de email"""
    
    def __init__(self, storage_path: Optional[str] = None):
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            self.storage_path = Path(__file__).parent / "accounts.json"
        
        self.client = MailTMClient()
        self.accounts: Dict[str, EmailAccount] = {}
        self._load_accounts()
    
    def _load_accounts(self):
        """Carrega contas do arquivo"""
        if self.storage_path.exists():
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                for addr, acc_data in data.items():
                    self.accounts[addr] = EmailAccount.from_dict(acc_data)
    
    def _save_accounts(self):
        """Salva contas no arquivo"""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {addr: acc.to_dict() for addr, acc in self.accounts.items()}
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def create_email(self, username: Optional[str] = None) -> EmailAccount:
        """Cria um novo email e salva"""
        account = self.client.create_account(address=username)
        self.accounts[account.address] = account
        self._save_accounts()
        return account
    
    def create_multiple_emails(self, count: int, prefix: Optional[str] = None, 
                                delay: float = 3.0, max_retries: int = 3) -> List[EmailAccount]:
        """Cria múltiplos emails de uma vez com retry"""
        created = []
        for i in range(count):
            retries = 0
            while retries < max_retries:
                try:
                    if prefix:
                        username = f"{prefix}{i+1}"
                    else:
                        username = None
                    account = self.create_email(username)
                    created.append(account)
                    print(f"  ✓ Criado: {account.address}")
                    time.sleep(delay)
                    break
                except Exception as e:
                    retries += 1
                    if "429" in str(e) and retries < max_retries:
                        wait_time = delay * (2 ** retries)  # Exponential backoff
                        print(f"  Rate limit, aguardando {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"  ✗ Erro ao criar email {i+1}: {e}")
                        break
        return created
    
    def get_all_accounts(self) -> List[EmailAccount]:
        """Retorna todas as contas"""
        return list(self.accounts.values())
    
    def get_account(self, address: str) -> Optional[EmailAccount]:
        """Retorna uma conta específica"""
        return self.accounts.get(address)
    
    def refresh_token(self, address: str) -> bool:
        """Atualiza o token de uma conta"""
        account = self.accounts.get(address)
        if not account:
            return False
        try:
            new_token = self.client.get_token(account.address, account.password)
            account.token = new_token
            self._save_accounts()
            return True
        except:
            return False
    
    def check_inbox(self, address: str) -> List[EmailMessage]:
        """Verifica inbox de uma conta"""
        account = self.accounts.get(address)
        if not account:
            return []
        
        try:
            messages = self.client.get_messages(account.token)
            account.last_checked = datetime.now().isoformat()
            self._save_accounts()
            return messages
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                if self.refresh_token(address):
                    account = self.accounts.get(address)
                    return self.client.get_messages(account.token)
            raise
    
    def check_all_inboxes(self) -> Dict[str, List[EmailMessage]]:
        """Verifica inbox de todas as contas"""
        results = {}
        for address in self.accounts:
            try:
                results[address] = self.check_inbox(address)
            except Exception as e:
                results[address] = []
                print(f"Erro ao verificar {address}: {e}")
        return results
    
    def get_message_details(self, address: str, message_id: str) -> Optional[EmailMessage]:
        """Retorna detalhes de uma mensagem"""
        account = self.accounts.get(address)
        if not account:
            return None
        return self.client.get_message(account.token, message_id)
    
    def delete_account(self, address: str) -> bool:
        """Deleta uma conta"""
        account = self.accounts.get(address)
        if not account:
            return False
        
        try:
            self.client.delete_account(account.token, account.id)
        except:
            pass
        
        del self.accounts[address]
        self._save_accounts()
        return True
    
    def export_accounts(self, filepath: str):
        """Exporta contas para arquivo"""
        data = []
        for acc in self.accounts.values():
            data.append({
                "email": acc.address,
                "password": acc.password,
                "created": acc.created_at
            })
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def wait_for_email(self, address: str, timeout: int = 300, interval: int = 5, 
                       subject_filter: Optional[str] = None) -> Optional[EmailMessage]:
        """Aguarda por um email específico"""
        account = self.accounts.get(address)
        if not account:
            return None
        
        start_time = time.time()
        seen_ids = set()
        
        while time.time() - start_time < timeout:
            messages = self.check_inbox(address)
            for msg in messages:
                if msg.id not in seen_ids:
                    seen_ids.add(msg.id)
                    if subject_filter:
                        if subject_filter.lower() in msg.subject.lower():
                            return self.get_message_details(address, msg.id)
                    else:
                        return self.get_message_details(address, msg.id)
            time.sleep(interval)
        
        return None


if __name__ == "__main__":
    manager = EmailManager()
    
    print("Criando email de teste...")
    account = manager.create_email()
    print(f"Email criado: {account.address}")
    print(f"Senha: {account.password}")
    
    print("\nVerificando inbox...")
    messages = manager.check_inbox(account.address)
    print(f"Mensagens: {len(messages)}")
