import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import rehypeRaw from "rehype-raw";

const Sobre = () => {
    const [content, setContent] = useState('');

    useEffect(() => {
        fetch("/README.md").then((response) => {
            response.text().then((text) => {
                setContent(text);
            });
        });
    }, []);

    return (
        <div style={{display: "flex", justifyContent: "center"}}>
            <div style={{maxWidth: "90vw", padding: "16px"}}>
                <ReactMarkdown rehypePlugins={[rehypeRaw]} children={content} skipHtml={false}/>
            </div>
        </div>
    )
}

export default Sobre;