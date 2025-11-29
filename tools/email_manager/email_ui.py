"""
Email Manager UI - Interface de terminal rica para gerenciar emails temporários
Usa Rich para uma experiência visual bonita no terminal
"""

import sys
import time
from typing import Optional
from datetime import datetime

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm, IntPrompt
    from rich.live import Live
    from rich.layout import Layout
    from rich.text import Text
    from rich.markdown import Markdown
    from rich import box
    from rich.progress import Progress, SpinnerColumn, TextColumn
except ImportError:
    print("Instalando dependência 'rich'...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm, IntPrompt
    from rich.live import Live
    from rich.layout import Layout
    from rich.text import Text
    from rich.markdown import Markdown
    from rich import box
    from rich.progress import Progress, SpinnerColumn, TextColumn

from email_core import EmailManager, EmailAccount, EmailMessage


console = Console()


def clear_screen():
    console.clear()


def show_header():
    header = """
╔═══════════════════════════════════════════════════════════════════╗
║            📧 EMAIL MANAGER - Emails Temporários 📧               ║
║              Powered by Mail.tm API (100% Gratuito)               ║
╚═══════════════════════════════════════════════════════════════════╝
"""
    console.print(header, style="bold cyan")


def show_menu():
    menu = Table(show_header=False, box=box.ROUNDED, border_style="cyan")
    menu.add_column("Opção", style="bold yellow", width=5)
    menu.add_column("Descrição", style="white")
    
    menu.add_row("1", "📧 Criar novo email")
    menu.add_row("2", "📧 Criar múltiplos emails")
    menu.add_row("3", "📋 Listar todos os emails")
    menu.add_row("4", "📬 Verificar inbox (todos)")
    menu.add_row("5", "📬 Verificar inbox (específico)")
    menu.add_row("6", "📖 Ler mensagem")
    menu.add_row("7", "⏳ Aguardar email")
    menu.add_row("8", "💾 Exportar contas")
    menu.add_row("9", "🗑️  Deletar conta")
    menu.add_row("0", "🚪 Sair")
    
    console.print(menu)


def show_accounts_table(accounts: list[EmailAccount]):
    if not accounts:
        console.print("\n[yellow]Nenhuma conta criada ainda.[/yellow]")
        return
    
    table = Table(title="📧 Contas de Email", box=box.DOUBLE_EDGE, border_style="green")
    table.add_column("#", style="dim", width=4)
    table.add_column("Email", style="cyan", no_wrap=True)
    table.add_column("Senha", style="yellow")
    table.add_column("Criado em", style="green")
    table.add_column("Último check", style="magenta")
    
    for i, acc in enumerate(accounts, 1):
        created = acc.created_at[:19].replace("T", " ") if acc.created_at else "-"
        last_check = acc.last_checked[:19].replace("T", " ") if acc.last_checked else "Nunca"
        table.add_row(str(i), acc.address, acc.password, created, last_check)
    
    console.print(table)


def show_inbox_table(address: str, messages: list[EmailMessage]):
    if not messages:
        console.print(f"\n[yellow]📭 Inbox vazio para {address}[/yellow]")
        return
    
    table = Table(title=f"📬 Inbox: {address}", box=box.ROUNDED, border_style="blue")
    table.add_column("#", style="dim", width=4)
    table.add_column("De", style="cyan", max_width=25)
    table.add_column("Assunto", style="yellow", max_width=40)
    table.add_column("Preview", style="white", max_width=30)
    table.add_column("Data", style="green", width=12)
    
    for i, msg in enumerate(messages, 1):
        date = msg.created_at[:10] if msg.created_at else "-"
        from_addr = msg.from_name or msg.from_address
        if len(from_addr) > 25:
            from_addr = from_addr[:22] + "..."
        subject = msg.subject[:40] if msg.subject else "(Sem assunto)"
        intro = msg.intro[:30] if msg.intro else ""
        table.add_row(str(i), from_addr, subject, intro, date)
    
    console.print(table)


def show_message_detail(msg: EmailMessage):
    console.print(Panel(
        f"[bold cyan]De:[/bold cyan] {msg.from_name} <{msg.from_address}>\n"
        f"[bold cyan]Assunto:[/bold cyan] {msg.subject}\n"
        f"[bold cyan]Data:[/bold cyan] {msg.created_at}\n"
        f"[bold cyan]Anexos:[/bold cyan] {'Sim' if msg.has_attachments else 'Não'}",
        title="📧 Detalhes da Mensagem",
        border_style="green"
    ))
    
    console.print(Panel(
        msg.text or msg.intro or "(Mensagem vazia)",
        title="📝 Conteúdo",
        border_style="blue"
    ))


def create_email(manager: EmailManager):
    clear_screen()
    show_header()
    
    console.print("\n[bold cyan]📧 Criar Novo Email[/bold cyan]\n")
    
    custom = Confirm.ask("Deseja definir um username customizado?", default=False)
    username = None
    if custom:
        username = Prompt.ask("Username (sem @domínio)")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Criando email...", total=None)
        try:
            account = manager.create_email(username)
            progress.update(task, completed=True)
        except Exception as e:
            console.print(f"\n[red]Erro: {e}[/red]")
            return
    
    console.print(Panel(
        f"[bold green]✅ Email criado com sucesso![/bold green]\n\n"
        f"[bold cyan]Email:[/bold cyan] {account.address}\n"
        f"[bold cyan]Senha:[/bold cyan] {account.password}",
        title="📧 Nova Conta",
        border_style="green"
    ))
    
    Prompt.ask("\nPressione Enter para continuar")


def create_multiple_emails(manager: EmailManager):
    clear_screen()
    show_header()
    
    console.print("\n[bold cyan]📧 Criar Múltiplos Emails[/bold cyan]\n")
    
    count = IntPrompt.ask("Quantos emails deseja criar?", default=5)
    prefix = Prompt.ask("Prefixo para os emails (opcional, ex: 'factory')", default="")
    
    if not prefix:
        prefix = None
    
    console.print(f"\n[yellow]Criando {count} emails...[/yellow]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task(f"Criando {count} emails...", total=count)
        
        created = []
        for i in range(count):
            try:
                if prefix:
                    username = f"{prefix}{i+1}"
                else:
                    username = None
                account = manager.create_email(username)
                created.append(account)
                progress.update(task, advance=1, description=f"Criado: {account.address}")
                time.sleep(0.3)
            except Exception as e:
                console.print(f"[red]Erro ao criar email {i+1}: {e}[/red]")
    
    console.print(f"\n[bold green]✅ {len(created)} emails criados com sucesso![/bold green]\n")
    show_accounts_table(created)
    
    Prompt.ask("\nPressione Enter para continuar")


def list_accounts(manager: EmailManager):
    clear_screen()
    show_header()
    
    accounts = manager.get_all_accounts()
    console.print(f"\n[bold cyan]📋 Total de contas: {len(accounts)}[/bold cyan]\n")
    show_accounts_table(accounts)
    
    Prompt.ask("\nPressione Enter para continuar")


def check_all_inboxes(manager: EmailManager):
    clear_screen()
    show_header()
    
    accounts = manager.get_all_accounts()
    if not accounts:
        console.print("\n[yellow]Nenhuma conta criada ainda.[/yellow]")
        Prompt.ask("\nPressione Enter para continuar")
        return
    
    console.print("\n[bold cyan]📬 Verificando todos os inboxes...[/bold cyan]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Verificando...", total=len(accounts))
        
        results = {}
        for acc in accounts:
            progress.update(task, description=f"Verificando: {acc.address}")
            try:
                messages = manager.check_inbox(acc.address)
                results[acc.address] = messages
            except Exception as e:
                results[acc.address] = []
                console.print(f"[red]Erro em {acc.address}: {e}[/red]")
            progress.advance(task)
    
    # Mostra resumo
    summary_table = Table(title="📊 Resumo dos Inboxes", box=box.ROUNDED, border_style="green")
    summary_table.add_column("Email", style="cyan")
    summary_table.add_column("Mensagens", style="yellow", justify="center")
    summary_table.add_column("Status", style="green")
    
    total_messages = 0
    for address, messages in results.items():
        count = len(messages)
        total_messages += count
        status = "📬 Tem emails!" if count > 0 else "📭 Vazio"
        summary_table.add_row(address, str(count), status)
    
    console.print(summary_table)
    console.print(f"\n[bold]Total de mensagens: {total_messages}[/bold]")
    
    # Pergunta se quer ver detalhes
    if total_messages > 0:
        if Confirm.ask("\nDeseja ver os detalhes das mensagens?", default=True):
            for address, messages in results.items():
                if messages:
                    show_inbox_table(address, messages)
    
    Prompt.ask("\nPressione Enter para continuar")


def check_specific_inbox(manager: EmailManager):
    clear_screen()
    show_header()
    
    accounts = manager.get_all_accounts()
    if not accounts:
        console.print("\n[yellow]Nenhuma conta criada ainda.[/yellow]")
        Prompt.ask("\nPressione Enter para continuar")
        return
    
    console.print("\n[bold cyan]📬 Verificar Inbox Específico[/bold cyan]\n")
    show_accounts_table(accounts)
    
    try:
        choice = IntPrompt.ask("\nEscolha o número da conta", default=1)
        if 1 <= choice <= len(accounts):
            account = accounts[choice - 1]
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task(f"Verificando {account.address}...", total=None)
                messages = manager.check_inbox(account.address)
            
            show_inbox_table(account.address, messages)
        else:
            console.print("[red]Escolha inválida![/red]")
    except Exception as e:
        console.print(f"[red]Erro: {e}[/red]")
    
    Prompt.ask("\nPressione Enter para continuar")


def read_message(manager: EmailManager):
    clear_screen()
    show_header()
    
    accounts = manager.get_all_accounts()
    if not accounts:
        console.print("\n[yellow]Nenhuma conta criada ainda.[/yellow]")
        Prompt.ask("\nPressione Enter para continuar")
        return
    
    console.print("\n[bold cyan]📖 Ler Mensagem[/bold cyan]\n")
    show_accounts_table(accounts)
    
    try:
        acc_choice = IntPrompt.ask("\nEscolha a conta", default=1)
        if not (1 <= acc_choice <= len(accounts)):
            console.print("[red]Escolha inválida![/red]")
            Prompt.ask("\nPressione Enter para continuar")
            return
        
        account = accounts[acc_choice - 1]
        messages = manager.check_inbox(account.address)
        
        if not messages:
            console.print(f"\n[yellow]📭 Inbox vazio para {account.address}[/yellow]")
            Prompt.ask("\nPressione Enter para continuar")
            return
        
        show_inbox_table(account.address, messages)
        
        msg_choice = IntPrompt.ask("\nEscolha a mensagem", default=1)
        if 1 <= msg_choice <= len(messages):
            msg = messages[msg_choice - 1]
            full_msg = manager.get_message_details(account.address, msg.id)
            if full_msg:
                clear_screen()
                show_header()
                show_message_detail(full_msg)
        else:
            console.print("[red]Escolha inválida![/red]")
    except Exception as e:
        console.print(f"[red]Erro: {e}[/red]")
    
    Prompt.ask("\nPressione Enter para continuar")


def wait_for_email(manager: EmailManager):
    clear_screen()
    show_header()
    
    accounts = manager.get_all_accounts()
    if not accounts:
        console.print("\n[yellow]Nenhuma conta criada ainda.[/yellow]")
        Prompt.ask("\nPressione Enter para continuar")
        return
    
    console.print("\n[bold cyan]⏳ Aguardar Email[/bold cyan]\n")
    show_accounts_table(accounts)
    
    try:
        acc_choice = IntPrompt.ask("\nEscolha a conta", default=1)
        if not (1 <= acc_choice <= len(accounts)):
            console.print("[red]Escolha inválida![/red]")
            return
        
        account = accounts[acc_choice - 1]
        timeout = IntPrompt.ask("Timeout em segundos", default=300)
        subject_filter = Prompt.ask("Filtrar por assunto (Enter para qualquer)", default="")
        
        if not subject_filter:
            subject_filter = None
        
        console.print(f"\n[yellow]Aguardando email em {account.address}...[/yellow]")
        console.print("[dim]Pressione Ctrl+C para cancelar[/dim]\n")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Aguardando novo email...", total=None)
            
            try:
                msg = manager.wait_for_email(
                    account.address, 
                    timeout=timeout, 
                    subject_filter=subject_filter
                )
                
                if msg:
                    progress.update(task, completed=True)
                    console.print("\n[bold green]✅ Email recebido![/bold green]\n")
                    show_message_detail(msg)
                else:
                    console.print("\n[yellow]⏰ Timeout - nenhum email recebido.[/yellow]")
            except KeyboardInterrupt:
                console.print("\n[yellow]Cancelado pelo usuário.[/yellow]")
    except Exception as e:
        console.print(f"[red]Erro: {e}[/red]")
    
    Prompt.ask("\nPressione Enter para continuar")


def export_accounts(manager: EmailManager):
    clear_screen()
    show_header()
    
    accounts = manager.get_all_accounts()
    if not accounts:
        console.print("\n[yellow]Nenhuma conta para exportar.[/yellow]")
        Prompt.ask("\nPressione Enter para continuar")
        return
    
    console.print("\n[bold cyan]💾 Exportar Contas[/bold cyan]\n")
    
    filepath = Prompt.ask("Nome do arquivo", default="emails_export.json")
    
    try:
        manager.export_accounts(filepath)
        console.print(f"\n[bold green]✅ Contas exportadas para: {filepath}[/bold green]")
    except Exception as e:
        console.print(f"[red]Erro: {e}[/red]")
    
    Prompt.ask("\nPressione Enter para continuar")


def delete_account(manager: EmailManager):
    clear_screen()
    show_header()
    
    accounts = manager.get_all_accounts()
    if not accounts:
        console.print("\n[yellow]Nenhuma conta para deletar.[/yellow]")
        Prompt.ask("\nPressione Enter para continuar")
        return
    
    console.print("\n[bold cyan]🗑️ Deletar Conta[/bold cyan]\n")
    show_accounts_table(accounts)
    
    try:
        choice = IntPrompt.ask("\nEscolha a conta para deletar", default=1)
        if 1 <= choice <= len(accounts):
            account = accounts[choice - 1]
            
            if Confirm.ask(f"[red]Tem certeza que deseja deletar {account.address}?[/red]", default=False):
                manager.delete_account(account.address)
                console.print(f"\n[green]✅ Conta {account.address} deletada.[/green]")
        else:
            console.print("[red]Escolha inválida![/red]")
    except Exception as e:
        console.print(f"[red]Erro: {e}[/red]")
    
    Prompt.ask("\nPressione Enter para continuar")


def main():
    manager = EmailManager()
    
    while True:
        clear_screen()
        show_header()
        
        # Mostra resumo rápido
        accounts = manager.get_all_accounts()
        console.print(f"\n[dim]Contas ativas: {len(accounts)}[/dim]\n")
        
        show_menu()
        
        choice = Prompt.ask("\n[bold]Escolha uma opção[/bold]", default="0")
        
        if choice == "1":
            create_email(manager)
        elif choice == "2":
            create_multiple_emails(manager)
        elif choice == "3":
            list_accounts(manager)
        elif choice == "4":
            check_all_inboxes(manager)
        elif choice == "5":
            check_specific_inbox(manager)
        elif choice == "6":
            read_message(manager)
        elif choice == "7":
            wait_for_email(manager)
        elif choice == "8":
            export_accounts(manager)
        elif choice == "9":
            delete_account(manager)
        elif choice == "0":
            console.print("\n[bold cyan]Até logo! 👋[/bold cyan]\n")
            break
        else:
            console.print("[red]Opção inválida![/red]")
            time.sleep(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n\n[bold cyan]Até logo! 👋[/bold cyan]\n")
