import { app } from "../../../scripts/app.js";
import { api } from '../../../scripts/api.js'
import { $el } from "../../../../scripts/ui.js";

function addStylesheet(url) {
    if (url.endsWith(".js")) {
        url = url.substr(0, url.length - 2) + "css";
    }
    $el("link", {
        parent: document.head,
        rel: "stylesheet",
        type: "text/css",
        href: url.startsWith("http") ? url : getUrl(url),
    });
}

function getUrl(path, baseUrl) {
    if (baseUrl) {
        return new URL(path, baseUrl).toString();
    } else {
        return new URL("../" + path, import.meta.url).toString();
    }
}

addStylesheet(getUrl("video.css", import.meta.url))

var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });

function getResourceURL(subfolder, filename, type = "input") {
    const params = [
        "filename=" + encodeURIComponent(filename),
        "type=" + type,
        "subfolder=" + subfolder,
        app.getRandParam().substring(1)
    ].join("&");
    return `/view?${params}`;
}
__name(getResourceURL, "getResourceURL");

function splitFilePath(path) {
    const folder_separator = path.lastIndexOf("/");
    if (folder_separator === -1) {
        return ["", path];
    }
    return [
        path.substring(0, folder_separator),
        path.substring(folder_separator + 1)
    ];
}
__name(splitFilePath, "splitFilePath");

async function uploadFile(videoWidget, videoUIWidget, file2, updateNode, pasted = false) {
    try {
        const body = new FormData();
        body.append("image", file2);
        if (pasted) body.append("subfolder", "DigtalHumen");
        const resp = await api.fetchApi("/upload/image", {
            method: "POST",
            body
        });
        if (resp.status === 200) {
            const data = await resp.json();
            let path = data.name;
            if (data.subfolder) path = data.subfolder + "/" + path;
            if (!videoWidget.options.values.includes(path)) {
                videoWidget.options.values.push(path);
            }
            if (updateNode) {
                videoUIWidget.element.src = api.apiURL(
                    getResourceURL(...splitFilePath(path))
                );
                videoWidget.value = path;
            }
        } else {
            alert(resp.status + " - " + resp.statusText);
        }
    } catch (error) {
        alert(error);
    }
}
__name(uploadFile, "uploadFile");


function fitHeight(node) {
    node.setSize([node.size[0], node.computeSize([node.size[0], node.size[1]])[1]])
    node?.graph?.setDirtyCanvas(true);
}

//注册视频预览
app.registerExtension({
    name: "ComfyUI_LatentSync.VideoWidget",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (["LoadVideo", "SaveLipSyncVideo"].includes(nodeType.comfyClass)) {
            nodeData.input.required.videoUI = ["VIDEO_UI"];
        }
    },
    getCustomWidgets() {
        return {
            VIDEO_UI(node, inputName) {
                const video = document.createElement("video");
                video.controls = true;
                video.loop = false;
                video.muted = false;
                video.classList.add("comfy-video");
                video.setAttribute("name", "media");
                const videoUIWidget = node.addDOMWidget(
                    inputName,
                    /* name=*/
                    "videoUI",
                    video,
                    {
                        serialize: false,
                    }
                );
                //默认不显示
                videoUIWidget.element.classList.add("empty-video-widget");

                //添加视频监听
                video.addEventListener('loadedmetadata', () => {
                    //如果有视频则显示
                    videoUIWidget.element.classList.remove("empty-video-widget");
                    //计算视频的比率
                    videoUIWidget.aspectRatio = videoUIWidget.element.videoWidth / videoUIWidget.element.videoHeight;
                    fitHeight(node);
                })
                //计算宽高
                videoUIWidget.computeSize = function (width) {
                    if (this.aspectRatio && !(videoUIWidget.element.classList || []).contains("empty-video-widget")) {
                        let height = (node.size[0] - 20) / this.aspectRatio + 10;
                        if (!(height > 0)) {
                            height = 0;
                        }
                        this.computedHeight = height + 10;
                        return [width, height];
                    }
                    return [width, -4]; //no loaded src, widget should not display
                }

                const isOutputNode = node.constructor.nodeData.output_node;
                if (isOutputNode) {
                    videoUIWidget.element.classList.add("empty-video-widget");
                    const onExecuted = node.onExecuted;
                    node.onExecuted = function (message) {
                        onExecuted?.apply(this, arguments);
                        const videos = message.videos;
                        if (!videos) return;
                        const video2 = videos[0];
                        videoUIWidget.element.src = api.apiURL(
                            getResourceURL(video2.subfolder, video2.filename, video2.type)
                        );
                        videoUIWidget.element.classList.remove("empty-video-widget");
                    };
                }
                return { widget: videoUIWidget };
            }
        };
    },
    onNodeOutputsUpdated(nodeOutputs) {
        for (const [nodeId, output] of Object.entries(nodeOutputs)) {
            const node = app.graph.getNodeById(nodeId);
            if ("video" in output) {
                const videoUIWidget = node.widgets.find(
                    (w) => w.name === "videoUI"
                );
                const video = output.video[0];
                videoUIWidget.element.src = api.apiURL(
                    getResourceURL(video.subfolder, video.filename, video.type)
                );
                videoUIWidget.element.classList.remove("empty-video-widget");
            }
        }
    },
});


app.registerExtension({
    name: "HBI-ComfyUI_LatentSync.UploadVideo",  // 扩展名称
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData?.input?.required?.video?.[1]?.video_upload === true) {
            nodeData.input.required.upload = ["VIDEOUPLOAD"];
        }
    },

    getCustomWidgets() {
        return {
            VIDEOUPLOAD(node, inputName) {
                //节点上的video框
                const videoWidget = node.widgets.find(
                    (w) => w.name === "video"
                );
                //节点上的预览视频
                const videoUIWidget = node.widgets.find(
                    (w) => w.name === "videoUI"
                );

                const onVideoWidgetUpdate = /* @__PURE__ */ __name(() => {
                    videoUIWidget.element.src = api.apiURL(
                        getResourceURL(...splitFilePath(videoWidget.value))
                    );
                }, "onVideoWidgetUpdate");

                if (videoWidget.value) {
                    onVideoWidgetUpdate();
                }

                videoWidget.callback = onVideoWidgetUpdate;
                const onGraphConfigured = node.onGraphConfigured;
                node.onGraphConfigured = function () {
                    onGraphConfigured?.apply(this, arguments);
                    if (videoWidget.value) {
                        onVideoWidgetUpdate();
                    }
                };
                const fileInput = document.createElement("input");
                fileInput.type = "file";
                fileInput.accept = "video/*";
                fileInput.style.display = "none";
                fileInput.onchange = () => {
                    if (fileInput.files.length) {
                        uploadFile(videoWidget, videoUIWidget, fileInput.files[0], true, true);
                    }
                };
                const uploadWidget = node.addWidget(
                    "button",
                    inputName,
                    /* value=*/
                    "",
                    () => {
                        fileInput.click();
                    },
                    { serialize: false }
                );
                uploadWidget.label = "请选择上传文件";
                return { widget: uploadWidget };
            }
        }
    }
});