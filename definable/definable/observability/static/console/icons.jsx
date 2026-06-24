// Stroke-only icon set. 16x16 default. Stroke 1.5.
// Every icon assigned to window.Icons.X for cross-file access.

const Icon = ({ children, size = 16, stroke = 1.5, className = '', style }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth={stroke}
    strokeLinecap="round"
    strokeLinejoin="round"
    className={className}
    style={style}
    aria-hidden="true"
  >
    {children}
  </svg>
);

const IconPlus = (p) => (<Icon {...p}><path d="M12 5v14M5 12h14" /></Icon>);
const IconArrowLeft = (p) => (<Icon {...p}><path d="M19 12H5M12 19l-7-7 7-7" /></Icon>);
const IconArrowUp = (p) => (<Icon {...p}><path d="M12 19V5M5 12l7-7 7 7" /></Icon>);
const IconChevronDown = (p) => (<Icon {...p}><path d="M6 9l6 6 6-6" /></Icon>);
const IconChevronRight = (p) => (<Icon {...p}><path d="M9 6l6 6-6 6" /></Icon>);
const IconChevronsRight = (p) => (<Icon {...p}><path d="M13 17l5-5-5-5M6 17l5-5-5-5" /></Icon>);
const IconSidebar = (p) => (<Icon {...p}><rect x="3" y="3" width="18" height="18" rx="2" /><path d="M9 3v18" /></Icon>);
const IconSearch = (p) => (<Icon {...p}><circle cx="11" cy="11" r="7" /><path d="M21 21l-4.3-4.3" /></Icon>);
const IconFilter = (p) => (<Icon {...p}><path d="M4 6h16M7 12h10M10 18h4" /></Icon>);
const IconShare = (p) => (<Icon {...p}><path d="M4 12v7a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-7M16 6l-4-4-4 4M12 2v14" /></Icon>);
const IconMore = (p) => (<Icon {...p}><circle cx="5" cy="12" r="1" /><circle cx="12" cy="12" r="1" /><circle cx="19" cy="12" r="1" /></Icon>);
const IconMoreV = (p) => (<Icon {...p}><circle cx="12" cy="5" r="1" /><circle cx="12" cy="12" r="1" /><circle cx="12" cy="19" r="1" /></Icon>);
const IconCopy = (p) => (<Icon {...p}><rect x="9" y="9" width="13" height="13" rx="2" /><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" /></Icon>);
const IconFolder = (p) => (<Icon {...p}><path d="M4 5a2 2 0 0 1 2-2h4l2 2h6a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V5z" /></Icon>);
const IconMail = (p) => (<Icon {...p}><rect x="3" y="5" width="18" height="14" rx="2" /><path d="M3 7l9 6 9-6" /></Icon>);
const IconHash = (p) => (<Icon {...p}><path d="M4 9h16M4 15h16M10 3L8 21M16 3l-2 18" /></Icon>);
const IconChat = (p) => (<Icon {...p}><path d="M21 12c0 4.4-4 8-9 8a9.7 9.7 0 0 1-4-.85L3 20l1.2-3.5A8.4 8.4 0 0 1 3 12c0-4.4 4-8 9-8s9 3.6 9 8z" /></Icon>);
const IconStar = (p) => (<Icon {...p}><path d="M12 2l3 7h7l-5.5 4.5L18 22l-6-4-6 4 1.5-8.5L2 9h7z" /></Icon>);
const IconSparkle = (p) => (<Icon {...p}><path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8L12 3z" /></Icon>);
const IconBolt = (p) => (<Icon {...p}><path d="M13 2L4 14h7l-2 8 9-12h-7l2-8z" /></Icon>);
const IconGrid = (p) => (<Icon {...p}><rect x="3" y="3" width="7" height="7" /><rect x="14" y="3" width="7" height="7" /><rect x="3" y="14" width="7" height="7" /><rect x="14" y="14" width="7" height="7" /></Icon>);
const IconCube = (p) => (<Icon {...p}><path d="M12 2l9 5v10l-9 5-9-5V7l9-5zM3 7l9 5 9-5M12 12v10" /></Icon>);
const IconGitBranch = (p) => (<Icon {...p}><circle cx="6" cy="6" r="2" /><circle cx="6" cy="18" r="2" /><circle cx="18" cy="6" r="2" /><path d="M6 8v8M18 8v2a4 4 0 0 1-4 4H8" /></Icon>);
const IconUsers = (p) => (<Icon {...p}><circle cx="9" cy="8" r="3.5" /><path d="M3 20c0-3 3-5 6-5s6 2 6 5M16 11a3 3 0 1 0 0-6M21 20c0-2.5-2-4-4-4.5" /></Icon>);
const IconGlobe = (p) => (<Icon {...p}><circle cx="12" cy="12" r="9" /><path d="M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18" /></Icon>);
const IconWeb = (p) => (<Icon {...p}><circle cx="12" cy="12" r="9" /><path d="M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18" /></Icon>);
const IconImage = (p) => (<Icon {...p}><rect x="3" y="3" width="18" height="18" rx="2" /><circle cx="9" cy="9" r="2" /><path d="M21 15l-5-5L5 21" /></Icon>);
const IconClock = (p) => (<Icon {...p}><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></Icon>);
const IconDocument = (p) => (<Icon {...p}><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><path d="M14 2v6h6M8 13h8M8 17h5" /></Icon>);
const IconDownload = (p) => (<Icon {...p}><path d="M12 3v12M6 11l6 6 6-6M5 21h14" /></Icon>);
const IconCal = (p) => (<Icon {...p}><rect x="3" y="5" width="18" height="16" rx="2" /><path d="M3 9h18M8 3v4M16 3v4" /></Icon>);
const IconPaperclip = (p) => (<Icon {...p}><path d="M21 11.5L12 21a5 5 0 0 1-7-7L14 5a3 3 0 0 1 4.5 4L10 17.5a1.5 1.5 0 0 1-2-2L16 7" /></Icon>);
const IconSkill = (p) => (<Icon {...p}><path d="M12 3l9 5-9 5-9-5 9-5zM3 13l9 5 9-5M3 18l9 5 9-5" /></Icon>);
const IconWebhook = (p) => (<Icon {...p}><circle cx="7" cy="7" r="3" /><circle cx="17" cy="17" r="3" /><path d="M9 9l6 6M10 14l-3 3M14 10l3-3" /></Icon>);
const IconSun = (p) => (<Icon {...p}><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></Icon>);
const IconMoon = (p) => (<Icon {...p}><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" /></Icon>);
const IconEdit = (p) => (<Icon {...p}><path d="M12 20h9" /><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4 12.5-12.5z" /></Icon>);
const IconCheck = (p) => (<Icon {...p}><path d="M5 12l5 5 9-11" /></Icon>);
const IconX = (p) => (<Icon {...p}><path d="M6 6l12 12M18 6L6 18" /></Icon>);

window.Icons = {
  Icon,
  IconPlus, IconArrowLeft, IconArrowUp, IconChevronDown, IconChevronRight, IconChevronsRight,
  IconSidebar, IconSearch, IconFilter, IconShare, IconMore, IconMoreV, IconCopy,
  IconFolder, IconMail, IconHash, IconChat, IconStar, IconSparkle, IconBolt,
  IconGrid, IconCube, IconGitBranch, IconUsers, IconGlobe, IconWeb, IconImage,
  IconClock, IconDocument, IconDownload, IconCal, IconPaperclip, IconSkill, IconWebhook,
  IconSun, IconMoon, IconEdit, IconCheck, IconX,
};
