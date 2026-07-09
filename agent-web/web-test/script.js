const API_URL = "http://127.0.0.1:8000/chat"; 

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
    const fileInput = document.getElementById("file-input");
    const input = document.getElementById("user-input");
    const text = input.value.trim();
    const file = fileInput.files[0];

    if(text === "" && !file) return;

    addMessage(text, "user");

    input.value = "";

    const loading = addLoading();

    try{
        const formData = new FormData();
        formData.append("message", text);

        if(file){
            formData.append("file", file);
        }

        const response = await fetch(API_URL,{
            method:"POST",
            body: formData
        });

        const data = await response.json();

        removeLoading(loading);

        if(data.type === "image"){
            addImage(data.response);
        }else{
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


//checkServerStatus();

//setInterval(checkServerStatus,10000);

// function uploadFile() {
//     const fileInput = document.getElementById("file-input");
//     const file = fileInput.files[0];
//     if (!file) return;

//     const formData = new FormData();
//     formData.append("file", file);

//     fetch("http://127.0.0.1:8000/chat", {
//         method: "POST",
//         body: formData
//     })
//     .then(response => response.json())
//     .then(data => {
//         addMessage("System", "File uploaded: " + file.name);
//     })
//     .catch(error => {
//         console.error("Error:", error);
//     });
// }