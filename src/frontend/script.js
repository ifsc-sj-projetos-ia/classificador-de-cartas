document.addEventListener('DOMContentLoaded', () => {
    // ---------- Tradução das classes (inglês do dataset -> PT-BR) ----------
    const VALUES = {
        ace: 'Ás', two: '2', three: '3', four: '4', five: '5', six: '6',
        seven: '7', eight: '8', nine: '9', ten: '10',
        jack: 'Valete', queen: 'Dama', king: 'Rei',
    };
    const SUITS = {
        clubs: { nome: 'Paus', simbolo: '♣', cor: 'preto' },
        diamonds: { nome: 'Ouros', simbolo: '♦', cor: 'vermelho' },
        hearts: { nome: 'Copas', simbolo: '♥', cor: 'vermelho' },
        spades: { nome: 'Espadas', simbolo: '♠', cor: 'preto' },
    };

    // "four of diamonds" -> { texto, simbolo, valorCurto, cor }
    function traduzCarta(classe) {
        if (classe === 'joker') {
            return { texto: 'Coringa', simbolo: '🃏', valorCurto: '★', cor: 'preto' };
        }
        const m = classe.match(/^(\w+) of (\w+)$/);
        if (!m) return { texto: classe, simbolo: '?', valorCurto: '?', cor: 'preto' };
        const valor = VALUES[m[1]] || m[1];
        const naipe = SUITS[m[2]] || { nome: m[2], simbolo: '?', cor: 'preto' };
        return {
            texto: `${valor} de ${naipe.nome}`,
            simbolo: naipe.simbolo,
            valorCurto: valor === 'Ás' ? 'A' : (VALUES[m[1]] && m[1] === 'ace' ? 'A' : (m[1] === 'jack' ? 'J' : m[1] === 'queen' ? 'Q' : m[1] === 'king' ? 'K' : valor)),
            cor: naipe.cor,
        };
    }

    // ---------- Navegação entre páginas ----------
    document.querySelectorAll('.topnav a').forEach(link => {
        link.addEventListener('click', e => {
            e.preventDefault();
            document.querySelectorAll('.topnav a').forEach(l => l.classList.remove('active'));
            document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
            link.classList.add('active');
            document.getElementById(link.dataset.target).classList.add('active');
        });
    });

    // ---------- Status do servidor ----------
    function pollStatus() {
        fetch('/api/status').then(r => r.json()).then(d => {
            const el = document.getElementById('initStatus');
            if (d.ready) {
                // Modelo carregado: o aviso some (nao polui a tela).
                el.classList.add('hidden');
            } else {
                el.textContent = d.message || 'Carregando modelo…';
                setTimeout(pollStatus, 1500);
            }
        }).catch(() => {});
    }
    pollStatus();

    // ---------- Página "Sobre o modelo": stats ----------
    fetch('/api/stats').then(r => r.json()).then(d => {
        document.getElementById('nClasses').textContent = d.n_classes ?? '—';
        document.getElementById('trainSize').textContent = (d.train_size || 0).toLocaleString('pt-BR');
        document.getElementById('valSize').textContent = d.val_size ?? '—';
        document.getElementById('testSize').textContent = d.test_size ?? '—';

        if (d.main_model) {
            document.getElementById('mainAcc').textContent = (d.main_model.test_acc * 100).toFixed(1) + '%';
            document.getElementById('mainF1').textContent = d.main_model.test_macroF1.toFixed(3);
        }
        document.getElementById('oodAcc').textContent = ((d.ood_acc || 0) * 100).toFixed(1) + '%';

        const expEl = document.getElementById('experimentsList');
        expEl.innerHTML = (d.experiments || []).map(e => `
            <div class="model-detail-row">
                <span class="model-name">${e.experimento}</span>
                <div class="acc-bar"><div style="width:${(e.test_acc * 100).toFixed(1)}%"></div></div>
                <span class="acc-value">${(e.test_acc * 100).toFixed(1)}%</span>
                <span class="f1-value">F1 ${e.test_macroF1.toFixed(3)}</span>
            </div>`).join('');

        const worstEl = document.getElementById('worstList');
        worstEl.innerHTML = (d.worst_classes || []).map(c => {
            const t = traduzCarta(c.classe);
            return `<div class="model-detail-row">
                <span class="model-name">${t.texto} <span class="muted">(${c.classe})</span></span>
                <div class="acc-bar"><div style="width:${(c.f1 * 100).toFixed(0)}%"></div></div>
                <span class="acc-value">F1 ${c.f1.toFixed(2)}</span>
            </div>`;
        }).join('') || '<p class="muted">Sem dados de relatório por classe.</p>';
    }).catch(() => {});

    // ---------- Upload / preview ----------
    const cameraInput = document.getElementById('cameraInput');
    const galleryInput = document.getElementById('galleryInput');
    const preview = document.getElementById('preview');
    const dropzoneInner = document.getElementById('dropzoneInner');
    const dropzone = document.getElementById('dropzone');
    const btnPredict = document.getElementById('btnPredict');
    let currentFile = null;

    function setFile(file) {
        if (!file || !file.type.startsWith('image/')) return;
        currentFile = file;
        const url = URL.createObjectURL(file);
        preview.src = url;
        preview.classList.remove('hidden');
        dropzoneInner.classList.add('hidden');
        btnPredict.disabled = false;
        document.getElementById('resultPanel').classList.add('hidden');
    }

    // Câmera (mobile abre a traseira) e galeria/arquivos disparam o mesmo fluxo.
    cameraInput.addEventListener('change', e => setFile(e.target.files[0]));
    galleryInput.addEventListener('change', e => setFile(e.target.files[0]));
    document.getElementById('btnCamera').addEventListener('click', () => cameraInput.click());
    document.getElementById('btnGallery').addEventListener('click', () => galleryInput.click());

    // ---------- Leitura em voz alta (acessibilidade) ----------
    // DESLIGADA por padrão; o usuário liga no botão e a escolha fica salva.
    const btnVoice = document.getElementById('btnVoice');
    let voiceOn = false;
    try { voiceOn = localStorage.getItem('voz') === 'on'; } catch (e) {}

    function refreshVoiceBtn() {
        btnVoice.setAttribute('aria-pressed', String(voiceOn));
        btnVoice.classList.toggle('active', voiceOn);
        // Rótulo fixo "Acessibilidade"; o estado fica no ícone/cor + aria-pressed.
        btnVoice.textContent = (voiceOn ? '🔊' : '🔈') + ' Acessibilidade';
    }
    // Se o navegador não tiver síntese de voz, esconde o botão.
    if (!('speechSynthesis' in window)) {
        btnVoice.classList.add('hidden');
    } else {
        refreshVoiceBtn();
        btnVoice.addEventListener('click', () => {
            voiceOn = !voiceOn;
            try { localStorage.setItem('voz', voiceOn ? 'on' : 'off'); } catch (e) {}
            window.speechSynthesis.cancel();
            refreshVoiceBtn();
        });
    }

    function falar(texto) {
        if (!voiceOn || !('speechSynthesis' in window)) return;
        window.speechSynthesis.cancel();
        const u = new SpeechSynthesisUtterance(texto);
        u.lang = 'pt-BR';
        window.speechSynthesis.speak(u);
    }

    ['dragover', 'dragenter'].forEach(ev =>
        dropzone.addEventListener(ev, e => { e.preventDefault(); dropzone.classList.add('drag'); }));
    ['dragleave', 'drop'].forEach(ev =>
        dropzone.addEventListener(ev, e => { e.preventDefault(); dropzone.classList.remove('drag'); }));
    dropzone.addEventListener('drop', e => {
        if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
    });


    // ---------- Predição ----------
    btnPredict.addEventListener('click', () => {
        if (!currentFile) return;
        const fd = new FormData();
        fd.append('image', currentFile);

        btnPredict.disabled = true;
        btnPredict.textContent = 'Reconhecendo…';

        fetch('/api/predict', { method: 'POST', body: fd })
            .then(r => r.json().then(j => ({ ok: r.ok, j })))
            .then(({ ok, j }) => {
                if (!ok) { alert(j.error || 'Erro na predição.'); return; }
                renderResult(j);
            })
            .catch(() => alert('Erro ao conectar. O servidor está rodando (python -m src.app)?'))
            .finally(() => {
                btnPredict.disabled = false;
                btnPredict.textContent = 'Reconhecer carta';
            });
    });

    function renderResult(data) {
        const panel = document.getElementById('resultPanel');
        panel.classList.remove('hidden');

        const t = traduzCarta(data.best.classe);
        const conf = (data.best.prob * 100).toFixed(1);

        const big = document.getElementById('bigCard');
        big.textContent = t.valorCurto + ' ' + t.simbolo;
        big.className = 'big-card ' + t.cor;

        document.getElementById('verdictText').textContent = t.texto;
        document.getElementById('confText').textContent = `${conf}% de confiança`;

        const list = document.getElementById('topkList');
        list.innerHTML = (data.top || []).map(p => {
            const tt = traduzCarta(p.classe);
            const pct = (p.prob * 100).toFixed(1);
            return `<div class="topk-row">
                <span class="topk-name">${tt.simbolo} ${tt.texto}</span>
                <div class="bar"><div style="width:${pct}%"></div></div>
                <span class="topk-pct">${pct}%</span>
            </div>`;
        }).join('');

        falar(`${t.texto}. ${conf} por cento de confiança.`);
        panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
});
