window.getRoleSignalDemoConfidence = function (predictedRole) {
    const role = String(predictedRole || '').trim().toLowerCase();

    if (/\b(hr|human resources|recruit|talent acquisition)\b/.test(role)) {
        return 87;
    }
    if (/\b(data|analytics|analyst|business intelligence)\b/.test(role)) {
        return 85;
    }
    if (/\b(software|python|developer|engineer|programmer|web)\b/.test(role)) {
        return 88;
    }
    if (!role) {
        return 84;
    }

    let hash = 0;
    for (let index = 0; index < role.length; index += 1) {
        hash = (hash * 31 + role.charCodeAt(index)) >>> 0;
    }
    return 80 + (hash % 11);
};
