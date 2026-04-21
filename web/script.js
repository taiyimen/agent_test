const API_URL = "http://localhost:8000/chat"; 

function getCurrentTime(){

    const now = new Date();

    const hours = String(now.getHours()).padStart(2,'0');
    const minutes = String(now.getMinutes()).padStart(2,'0');

    return hours + ":" + minutes;
}

function addMessage(text, sender){

    const chatBox = document.getElementById("chat-box");

    const msg = document.createElement("div");
    msg.className = "message " + sender;

    const time = document.createElement("div");
    time.className = "time";
    time.innerText = getCurrentTime();

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.innerText = text;

    msg.appendChild(time);
    msg.appendChild(bubble);

    chatBox.appendChild(msg);

    chatBox.scrollTop = chatBox.scrollHeight;
}
function handleKey(event){
    if(event.key === "Enter"){
        sendMessage();
    }
}

async function sendMessage(){

    const input = document.getElementById("user-input");
    const text = input.value.trim();

    if(text === "") return;

    addMessage(text, "user");

    input.value = "";

    const loading = addLoading();

    try{

        const response = await fetch(API_URL,{
            method:"POST",
            headers:{
                "Content-Type":"application/json"
            },
            body:JSON.stringify({
                message:text
            })
        });

        const data = await response.json();

        removeLoading(loading);

        if(data.type === "image"){
            addImage(data.response);

        } else if (data.type === "shap_table") {
            renderShapTable(data);

        } else {
            addMessage(data.response,"bot");
        }

    }catch(err){

        removeLoading(loading);

        addMessage("服务器连接失败","bot");

    }

}

function addLoading(){

    const chatBox = document.getElementById("chat-box");

    const msg = document.createElement("div");
    msg.className = "message bot";

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.innerText = "思考中...";

    msg.appendChild(bubble);
    chatBox.appendChild(msg);

    return msg;
}

function removeLoading(element){
    element.remove();
}

function addImage(src){

    const chatBox = document.getElementById("chat-box");

    const msg = document.createElement("div");
    msg.className = "message bot";

    const time = document.createElement("div");
    time.className = "time";
    time.innerText = getCurrentTime();

    const bubble = document.createElement("div");
    bubble.className = "bubble";

    const img = document.createElement("img");
    img.src = src;
    img.style.maxWidth = "400px";

    bubble.appendChild(img);

    msg.appendChild(time);
    msg.appendChild(bubble);

    chatBox.appendChild(msg);

    chatBox.scrollTop = chatBox.scrollHeight;
}

function setStatus(status){

    const el = document.querySelector(".status-indicator");

    if(status){
        el.innerText="在线";
        el.className="status-indicator online";
    }else{
        el.innerText="离线";
        el.className="status-indicator offline";
    }
}

function checkServerStatus(){

    fetch(API_URL,{
        method:"POST",
        headers:{
            "Content-Type":"application/json"
        },
        body:JSON.stringify({message:"ping"})
    })
    .then(res=>{
        if(res.ok){
            document.querySelector(".status-indicator").innerText="Online";
        }else{
            document.querySelector(".status-indicator").innerText="Offline";
        }
    })
    .catch(()=>{
        document.querySelector(".status-indicator").innerText="Offline";
    });

}

window.onload = function () {
    // 使用统一的 addMessage 函数，确保样式和后续对话一致
    const welcomeText = `Hello! 👋 I am an intelligent agent developed by the expert team at Peking University International Hospital for predicting postoperative complications after retroperitoneal tumor resection and supporting clinical decision-making.

Currently, I support analysis of acute kidney injury (AKI) after retroperitoneal tumor surgery, and additional features are under development.

I can help you with:
• Assessing the risk of postoperative acute kidney injury (AKI)
• Interpreting clinical indicators
• Visualizing results

Please enter the patient’s relevant information. Image and file inputs are also supported, and I will provide prediction and analysis for you.
`;

    addMessage(welcomeText, "bot");
};


checkServerStatus();

setInterval(checkServerStatus,10000);

function uploadFile() {
    const fileInput = document.getElementById("file-input");
    const file = fileInput.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData
    })
    .then(async response => {
        let text;
        try {
            const data = await response.json();
            if (response.ok && data && data.success) {
                text = `文件上传成功: ${data.filename}`;
            } else {
                text = `上传失败: ${response.status} ${response.statusText} - ${JSON.stringify(data)}`;
            }
        } catch (e) {
            text = `上传返回非 JSON 响应: ${response.status} ${response.statusText}`;
        }
        addMessage(text, "bot");
    })
    .catch(error => {
        console.error("Upload error:", error);
        addMessage("文件上传出错：" + error.message, "bot");
    });
}

function addHTMLMessage(html, sender){
    const chatBox = document.getElementById("chat-box");

    const msg = document.createElement("div");
    msg.className = "message " + sender;

    const time = document.createElement("div");
    time.className = "time";
    time.innerText = getCurrentTime();

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.innerHTML = html;

    msg.appendChild(time);
    msg.appendChild(bubble);

    chatBox.appendChild(msg);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function renderShapTable(data){
    // data.shap_sorted: [{feature: f, shap: v}, ...]
    const list = data.shap_sorted || [];
    if(list.length === 0){
        addMessage('未返回 SHAP 结果','bot');
        return;
    }

    const rows = list.map(item =>{
        const name = item.feature;
        const val = Number(item.shap);
        const abs = Math.abs(val);
        let comment = '';
        if(val > 0){
            if(abs > 1) comment = '（显著增加风险）';
            else if(abs > 0.4) comment = '（增加风险）';
            else if(abs > 0.1) comment = '（轻微增加风险）';
            else comment = '（影响较小）';
        } else if (val < 0){
            if(abs > 1) comment = '（显著降低风险）';
            else if(abs > 0.4) comment = '（降低风险）';
            else if(abs > 0.1) comment = '（轻微降低风险）';
            else comment = '（影响较小）';
        } else {
            comment = '（无明显影响）';
        }
        return `<li><strong>${name}</strong>: ${val.toFixed(3)} ${comment}</li>`;
    }).join('');

    const html = `<div><strong>特征贡献（按 |SHAP| 降序）</strong><ul style="margin-top:6px">${rows}</ul></div>`;
    addHTMLMessage(html, 'bot');
}